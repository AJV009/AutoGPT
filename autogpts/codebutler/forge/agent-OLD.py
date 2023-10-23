import json
import pprint
from typing import List
import openai

from forge.sdk import (
    Agent,
    AgentDB,
    Step,
    StepRequestBody,
    Workspace,
    ForgeLogger,
    Task,
    TaskRequestBody,
    PromptEngine,
    chat_completion_request, Status,
)

LOG = ForgeLogger(__name__)

class ForgeAgent(Agent):
    """
    The goal of the Forge is to take care of the boilerplate code, so you can focus on
    agent design.

    There is a great paper surveying the agent landscape: https://arxiv.org/abs/2308.11432
    Which I would highly recommend reading as it will help you understand the possabilities.

    Here is a summary of the key components of an agent:

    Anatomy of an agent:
         - Profile
         - Memory
         - Planning
         - Action

    Profile:

    Agents typically perform a task by assuming specific roles. For example, a teacher,
    a coder, a planner etc. In using the profile in the llm prompt it has been shown to
    improve the quality of the output. https://arxiv.org/abs/2305.14688

    Additionally, based on the profile selected, the agent could be configured to use a
    different llm. The possibilities are endless and the profile can be selected
    dynamically based on the task at hand.

    Memory:

    Memory is critical for the agent to accumulate experiences, self-evolve, and behave
    in a more consistent, reasonable, and effective manner. There are many approaches to
    memory. However, some thoughts: there is long term and short term or working memory.
    You may want different approaches for each. There has also been work exploring the
    idea of memory reflection, which is the ability to assess its memories and re-evaluate
    them. For example, condensing short term memories into long term memories.

    Planning:

    When humans face a complex task, they first break it down into simple subtasks and then
    solve each subtask one by one. The planning module empowers LLM-based agents with the ability
    to think and plan for solving complex tasks, which makes the agent more comprehensive,
    powerful, and reliable. The two key methods to consider are: Planning with feedback and planning
    without feedback.

    Action:

    Actions translate the agent's decisions into specific outcomes. For example, if the agent
    decides to write a file, the action would be to write the file. There are many approaches you
    could implement actions.

    The Forge has a basic module for each of these areas. However, you are free to implement your own.
    This is just a starting point.
    """

    def __init__(self, database: AgentDB, workspace: Workspace):
        """
        The database is used to store tasks, steps and artifact metadata. The workspace is used to
        store artifacts. The workspace is a directory on the file system.

        Feel free to create subclasses of the database and workspace to implement your own storage
        """
        super().__init__(database, workspace)

        # Switch model to the latest GPT-4 model if available
        self.MODEL_NAME = "gpt-3.5-turbo"
        openai_supported_models = openai.Model.list()["data"]
        for model in openai_supported_models:
            if model["id"].startswith("gpt-4"):
                self.MODEL_NAME = "gpt-4"
                break
        LOG.info(f"agent.py - __init__ - Using model {self.MODEL_NAME}")

    async def create_task(self, task_request: TaskRequestBody) -> Task:
        """
        The agent protocol, which is the core of the Forge, works by creating a task and then
        executing steps for that task. This method is called when the agent is asked to create
        a task.

        We are hooking into function to add a custom log message. Though you can do anything you
        want here.
        """
        task = await super().create_task(task_request)
        LOG.info(
            f"agent.py - create_task - Task created: {task.task_id} input: {task.input[:40]}{'...' if len(task.input) > 40 else ''}"
        )
        return task

    async def execute_step(self, task_id: str, step_request: StepRequestBody, is_retry: bool = False) -> Step:
        task = await self.db.get_task(task_id)
        LOG.info(f"agent.py - execute_step - Execute step for task {task.task_id} input: {task.input[:19]}")

        steps, page = await self.db.list_steps(task_id, per_page=100)
        if not steps:
            LOG.info(f"agent.py - execute_step - No steps found, create plan steps")
            return await self.plan_steps(task, step_request)

        LOG.info(f"agent.py - execute_step - Found {len(steps)} steps")
        previous_steps = []
        next_steps = []
        for step in steps:
            if step.status == Status.created:
                LOG.info(f"agent.py - execute_step - Found created step {step.step_id} input: {step.input[:19]}")
                next_steps.append(step)
            elif step.status == Status.completed:
                LOG.info(f"agent.py - execute_step - Found completed step {step.step_id} input: {step.input[:19]}")
                previous_steps.append(step)

        if not next_steps:
            LOG.info(f"agent.py - execute_step - Tried to execute with no next steps, return last step as the last")
            step = previous_steps[-1]
            step.is_last = True
            return step

        current_step = next_steps[0]
        next_steps = next_steps[1:]
        ability = current_step.additional_input["ability"]
        LOG.info(f"agent.py - execute_step - Found next step {current_step.step_id} input: {current_step.input[:19]} ability: {ability['name']}")

        ability = await self.review_ability(ability, previous_steps)

        if ability["name"] == "finish":
            LOG.info(f"Finish task")
            current_step.is_last = True
        else:
            LOG.info(f"agent.py - execute_step - Run ability {ability['name']} with arguments {ability['args']}")
            output = await self.abilities.run_ability(
                task_id, ability["name"], **ability["args"]
            )

            current_step.output = str(output)
            LOG.debug(f"agent.py - execute_step - Executed step [{current_step.name}] output:\n{current_step.output}")

            prompt_engine = PromptEngine("review-steps")

            system_kwargs = {
                "abilities": self.abilities.list_abilities_for_prompt(),
                "files": self.workspace.list(task.task_id, "/")
            }

            system_prompt = prompt_engine.load_prompt("system-prompt", **system_kwargs)
            system_format = prompt_engine.load_prompt("step-format")

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": system_format},
            ]

            LOG.debug(f"agent.py - execute_step - Will review {len(next_steps)} next steps ({len(previous_steps)} steps have been completed):")

            next_step_dicts = [{"name": step.name,
                                "description": step.input,
                                "ability": step.additional_input["ability"]
                                } for step in next_steps]
            if next_step_dicts:
                next_steps_json = json.dumps(next_step_dicts)
                LOG.debug(f"agent.py - execute_step - Next steps:\n{next_steps_json}")
            else:
                next_steps_json = None

            task_kwargs = {
                "task": task.input,
                "step": current_step,
                "next_steps": next_steps_json,
                "previous_steps": previous_steps
            }

            task_prompt = prompt_engine.load_prompt("user-prompt", **task_kwargs)
            messages.append({"role": "user", "content": task_prompt})

            answer = await self.do_steps_request(messages, new_plan=False, retry=5)
            LOG.debug(f"agent.py - execute_step - Answer:\n{pprint.pformat(answer)}")

            if "steps" in answer and answer["steps"]:
                if not isinstance(answer["steps"], list):
                    LOG.info(f"agent.py - execute_step - Invalid next steps provided {answer['steps']}")
                else:
                    #if len(answer["steps"]) == len(next_steps):
                    #    existing_abilities = [step.additional_input["ability"] for step in next_steps]
                    #    new_abilities = [step["ability"] for step in answer["steps"]]
                    #    if existing_abilities == new_abilities:
                    #        LOG.info(f"The abilities in the new steps are the same as the existing steps, skip replace")
                    #else:
                    LOG.info(f"agent.py - execute_step - Replace {len(next_steps)} steps with {len(answer['steps'])} new steps")
                    for next_step in next_steps:
                        await self.db.update_step(task.task_id, next_step.step_id, "skipped")

                    next_steps = []
                    for i, new_step in enumerate(answer["steps"]):
                        LOG.info(f"agent.py - execute_step - Create step {i + 1} {new_step['name']}:\n{new_step['ability']}")
                        await self.create_step(task.task_id, new_step)
                        next_steps.append(new_step)
            else:
                LOG.info(f"agent.py - execute_step - No new steps provided")

        await self.db.update_step(task.task_id, current_step.step_id, "completed", output=current_step.output)
        LOG.info(f"agent.py - execute_step - Step completed: {current_step.step_id} input: {current_step.input[:19]}")

        if not next_steps:
            LOG.info(f"agent.py - execute_step - Task completed: {task.task_id} input: {task.input[:19]}")
            current_step.is_last = True
        elif len(previous_steps) > 15:
            LOG.info(f"agent.py - execute_step - Giving up after {len(previous_steps)} steps")
            current_step.is_last = True

        LOG.info(f"agent.py - execute_step - Current step {current_step}")
        return current_step

    async def plan_steps(self, task, step_request: StepRequestBody):
        step_request.name = "Plan steps"
        step_request.input = step_request.input or "Create steps to accomplish the objective"
        LOG.debug(f"agent.py - plan_steps - Create step {step_request.name}:\n{step_request.input}")

        step = await self.db.create_step(task_id=task.task_id, input=step_request, is_last=False)
        LOG.debug(f"agent.py - plan_steps - Created step {step}")

        files = self.workspace.list(task.task_id, "/")
        LOG.debug(f"agent.py - plan_steps - Files:\n{pprint.pformat(files)}")

        prompt_engine = PromptEngine("plan-steps")
        task_kwargs = {
            "abilities": self.abilities.list_abilities_for_prompt(),
            "files": files
        }
        system_prompt = prompt_engine.load_prompt("system-prompt",  **task_kwargs)
        system_format = prompt_engine.load_prompt("step-format")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": system_format},
        ]

        task_kwargs = {
            "task": task.input,
        }
        task_prompt = prompt_engine.load_prompt("user-prompt",  **task_kwargs)
        messages.append({"role": "user", "content": task_prompt})

        answer = await self.do_steps_request(messages, new_plan=True)
        LOG.debug(f"agent.py - plan_steps - Answer:\n{pprint.pformat(answer)}")

        await self.create_steps(task.task_id, answer["steps"])
        await self.db.update_step(task.task_id, step.step_id, "completed", output=answer["thoughts"]["text"])

        LOG.info(f"agent.py - plan_steps - Step being returned: {step}")
        return step

    async def do_steps_request(self, messages: List[dict], new_plan: bool = False, retry: int = 0):
        chat_completion_kwargs = {
            "messages": messages,
            "model": self.MODEL_NAME,
        }
        LOG.info(f"agent.py - do_steps_request - Messages:\n{pprint.pformat(messages)}")

        async def do_retry(retry_messages: List[dict]):
            LOG.info(f"agent.py - do_steps_request - do_retry - Retry {retry}")
            if retry < 2:
                messages.extend(retry_messages)
                LOG.info(f"agent.py - do_steps_request - do_retry - Retry {retry} with messages:\n{pprint.pformat(messages)}")
                return await self.do_steps_request(messages, new_plan, retry=retry + 1)
            else:
                LOG.info(f"agent.py - do_steps_request - do_retry - Retry limit reached, aborting")
                raise Exception("Failed to create steps")

        try:
            #LOG.debug(pprint.pformat(messages))
            chat_response = await chat_completion_request(**chat_completion_kwargs)
            response = chat_response["choices"][0]["message"]["content"]
            answer = json.loads(chat_response["choices"][0]["message"]["content"])
            LOG.info(f"agent.py - do_steps_request - Response:\n{pprint.pformat(response)}")
            LOG.info(f"agent.py - do_steps_request - Answer:\n{pprint.pformat(answer)}")
            #LOG.debug(pprint.pformat(answer))
        except json.JSONDecodeError as e:
            LOG.error(f"agent.py - do_steps_request - Unable to parse chat response: {response}. Got exception {e}")
            return await do_retry([{"role": "user", "content": f"Invalid response. {e}. Please try again."}])
        except Exception as e:
            LOG.error(f"agent.py - do_steps_request - Unable to generate chat response: {e}")
            raise e

        if new_plan and "steps" not in answer and not answer["steps"]:
            LOG.info(f"agent.py - do_steps_request - No steps provided, retry {retry}")
            return await do_retry([{"role": "user", "content": "You must provide at least one step."}])

        for step in answer["steps"]:
            LOG.info(f"agent.py - do_steps_request - Validate step {step['name']}")
            invalid_abilities = self.validate_ability(step)
            if invalid_abilities:
                LOG.info(f"agent.py - do_steps_request - Invalid abilities: {invalid_abilities}")
                return await do_retry(messages)

        if "thoughts" in answer and answer["thoughts"]:
            debug_string = ""
            if "reasoning" in answer["thoughts"]:
                debug_string += f"\n\tReasoning: {answer['thoughts']['reasoning']}"
            if "criticism" in answer["thoughts"]:
                debug_string += f"\n\tCriticism: {answer['thoughts']['criticism']}"
            if "text" in answer["thoughts"]:
                debug_string += f"\n\tText: {answer['thoughts']['text']}"
            if "speak" in answer["thoughts"]:
                debug_string += f"\n\tSpeak: {answer['thoughts']['speak']}"
            LOG.info(f"agent.py - do_steps_request - Thoughts:{debug_string}")
        else:
            LOG.info(f"agent.py - do_steps_request - No thoughts provided, retry {retry}")

        return answer

    async def review_ability(self, ability, previous_steps):
        prompt_engine = PromptEngine("run-ability")
        system_kwargs = {
            "abilities": self.abilities.list_abilities_for_prompt(),
            "previous_steps": previous_steps
        }
        LOG.info(f"agent.py - review_ability - System kwargs:\n{pprint.pformat(system_kwargs)}")
        system_prompt = prompt_engine.load_prompt("system-prompt", **system_kwargs)
        ability_kwargs = {
            "ability": json.dumps(ability),
            "previous_steps": previous_steps
        }
        LOG.info(f"agent.py - review_ability - Ability kwargs:\n{pprint.pformat(ability_kwargs)}")
        ability_prompt = prompt_engine.load_prompt("user-prompt", **ability_kwargs)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ability_prompt},
        ]
        chat_completion_kwargs = {
            "messages": messages,
            "model": self.MODEL_NAME,
        }
        try:
            chat_response = await chat_completion_request(**chat_completion_kwargs)
            LOG.info(f"agent.py - review_ability - Chat response:\n{pprint.pformat(chat_response)}")
            ability_answer = json.loads(chat_response["choices"][0]["message"]["content"])
            ability_names = [a.name for a in self.abilities.list_abilities().values()]
            LOG.info(f"agent.py - review_ability - Ability answer:\n{pprint.pformat(ability_answer)}")
            if isinstance(ability_answer, dict) and ability_answer["name"] in ability_names:
                LOG.info(f"agent.py - review_ability - Valid ability: {ability_answer}")
                if ability != ability_answer:
                    LOG.info(f"agent.py - review_ability - Update ability {ability['name']} with {ability_answer['name']}")
                ability = ability_answer
            else:
                LOG.info(f"agent.py - review_ability - Invalid ability: {ability_answer}")

        except json.JSONDecodeError as e:
            LOG.warning(f"agent.py - review_ability - Unable to parse chat response: {chat_response}. Error: {e}.")
        except Exception as e:
            LOG.warning(f"agent.py - review_ability - Unable to generate chat response: {e}")
        return ability

    def validate_ability(self, step: dict):
        ability_names = [a.name for a in self.abilities.list_abilities().values()]
        LOG.info(f"agent.py - validate_ability - Valid abilities:\n{pprint.pformat(ability_names)}")
        invalid_abilities = []
        if "ability" not in step or not step["ability"]:
            LOG.info(f"agent.py - validate_ability - No ability found in step {step['name']}")
            invalid_abilities.append(f"No ability found in step {step['name']}")
        elif not isinstance(step["ability"], dict):
            LOG.info(f"agent.py - validate_ability - The ability in step {step['name']} was defined as a dictionary")
            invalid_abilities.append(f"The ability in step {step['name']} was defined as a dictionary")
        elif step["ability"]["name"] not in ability_names:
            LOG.info(f"agent.py - validate_ability - Ability {step['ability']['name']} in step {step['name']} does not exist, valid abilities are: {ability_names}")
            invalid_abilities.append(f"Ability {step['ability']['name']} in step {step['name']} does not exist, "
                                     f"valid abilities are: {ability_names}")
        LOG.info(f"agent.py - validate_ability - Invalid abilities:\n{pprint.pformat(invalid_abilities)}")
        return invalid_abilities

    async def create_steps(self, task_id: str, steps: list[dict]):
        for i, step in enumerate(steps):
            LOG.info(f"agent.py - create_steps - Create step {i+1} {step['name']}:\n{step['description']}\n{step['ability']}")
            await self.create_step(task_id, step)

    async def create_step(self, task_id: str, step: dict):
        step_request = StepRequestBody(
            name=step["name"],
            input=step["description"],
        )
        LOG.info(f"agent.py - create_step - Create step {step_request.name}:\n{step_request.input}")

        created_step = await self.db.create_step(
            task_id=task_id,
            input=step_request,
            additional_input={"ability": step["ability"]}
        )
        LOG.info(f"agent.py - create_step - Created step {created_step}")
        return created_step
