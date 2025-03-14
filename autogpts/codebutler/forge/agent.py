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

    async def execute_step(self, task_id: str, step_request: StepRequestBody) -> Step:
        """
        For a tutorial on how to add your own logic please see the offical tutorial series:
        https://aiedge.medium.com/autogpt-forge-e3de53cc58ec

        The agent protocol, which is the core of the Forge, works by creating a task and then
        executing steps for that task. This method is called when the agent is asked to execute
        a step.

        The task that is created contains an input string, for the benchmarks this is the task
        the agent has been asked to solve and additional input, which is a dictionary and
        could contain anything.

        If you want to get the task use:

        ```
        task = await self.db.get_task(task_id)
        ```

        The step request body is essentially the same as the task request and contains an input
        string, for the benchmarks this is the task the agent has been asked to solve and
        additional input, which is a dictionary and could contain anything.

        You need to implement logic that will take in this step input and output the completed step
        as a step object. You can do everything in a single step or you can break it down into
        multiple steps. Returning a request to continue in the step output, the user can then decide
        if they want the agent to continue or not.
        """
        # An example that
        task = await self.db.get_task(task_id)
        LOG.info(f"agent.py - execute_step - Execute step for task {task.task_id} input: {task.input[:19]}")
        
        steps, page = await self.db.list_steps(task_id, per_page=100)
        if not steps:
            LOG.info(f"agent.py - execute_step - No steps found, create plan steps")
            return await self.plan_steps(task, step_request)
        
        # step = await self.db.create_step(
        #     task_id=task_id, input=step_request, is_last=True
        # )

        # self.workspace.write(task_id=task_id, path="output.txt", data=b"Washington D.C")

        # await self.db.create_artifact(
        #     task_id=task_id,
        #     step_id=step.step_id,
        #     file_name="output.txt",
        #     relative_path="",
        #     agent_created=True,
        # )

        # step.output = "Washington D.C"

        # LOG.info(f"\t✅ Final Step completed: {step.step_id}. \n" +
        #          f"Output should be placeholder text Washington D.C. You'll need to \n" +
        #          f"modify execute_step to include LLM behavior. Follow the tutorial " +
        #          f"if confused. ")

        # return step

    async def plan_steps(self, task, step_request: StepRequestBody):

        # Plan Step Preparation
        step_request.name = "Plan steps"
        step_request.input = step_request.input or "Create steps to accomplish the objective"
        LOG.debug(f"agent.py - plan_steps - Create step {step_request.name}:\n{step_request.input}")
        step = await self.db.create_step(task_id=task.task_id, input=step_request, is_last=False)
        LOG.debug(f"agent.py - plan_steps - Created step {step}")
        files = self.workspace.list(task.task_id, "/")
        LOG.debug(f"agent.py - plan_steps - Files:\n{pprint.pformat(files)}")

        # DEBUG: Step 1: Generate Profile Selection Array
        profile_agent_gen_prompt_engine = PromptEngine("codebutler/profiles/agent-gen")
        task_kwargs = {
            "task": task.input,
            "tools": self.abilities.list_abilities_for_prompt(),
            "files": files,
        }
        messages = profile_agent_gen_prompt_engine.singular_message_array_generator(**task_kwargs)

        # DEBUG: Step 2: Generate Profile from LLM
        chat_completion_kwargs = {
            "messages": messages,
            "model": self.MODEL_NAME,
        }
        LOG.info(f"agent.py - do_steps_request - Messages:\n{pprint.pformat(messages)}")
        chat_response = await chat_completion_request(**chat_completion_kwargs)
        response = chat_response["choices"][0]["message"]["content"]
        LOG.info(f"agent.py - do_steps_request - Response:\n{pprint.pformat(response)}")
        answer = json.loads(chat_response["choices"][0]["message"]["content"])
        role = answer["role"]
        LOG.info(f"agent.py - do_steps_request - Role:\n{pprint.pformat(role)}")
        description = answer["description"]
        LOG.info(f"agent.py - do_steps_request - Description:\n{pprint.pformat(description)}")

        # DEBUG: Step 3: Generate Init Plan from Profile
        # Vars: role, role_description, tools, files, task
        task_kwargs["role"] = role
        task_kwargs["role_description"] = description
        plan_gen_prompt_engine = PromptEngine("codebutler/planning/plan-gen")
        messages = plan_gen_prompt_engine.singular_message_array_generator(**task_kwargs)

        # DEBUG: Step 4: Generate Plan from Init Plan
        chat_completion_kwargs = {
            "messages": messages,
            "model": self.MODEL_NAME,
        }
        chat_response = await chat_completion_request(**chat_completion_kwargs)
        response = chat_response["choices"][0]["message"]["content"]
        LOG.info(f"agent.py - do_steps_request - Response:\n{pprint.pformat(response)}")
        answer = json.loads(chat_response["choices"][0]["message"]["content"])
        plan_name = answer["plan_name"]
        LOG.info(f"agent.py - do_steps_request - Plan Name:\n{pprint.pformat(plan_name)}")
        execution_plan = answer["execution_plan"]
        LOG.info(f"agent.py - do_steps_request - Execution Plan:\n{pprint.pformat(execution_plan)}")
        variables = answer["variables"]
        LOG.info(f"agent.py - do_steps_request - Variables:\n{pprint.pformat(variables)}")
