from typing import Any, Dict

from spine.spine import SPINE

from prism.data import graph_sim


class PlanningSim:
    def __init__(self, debug=True):
        self.debug = debug

    def query_planner(self, llm_input: str, planner: SPINE) -> None:
        """Send a prompt to the SPINE planner and parse the response into a structured plan.

        Returns the parsed response dict (with 'plan' as a list of (action, arg) tuples)
        on success, or an empty dict if the planner fails after all retries.
        """
        resp, success, logs = planner.request(llm_input)

        if success:
            if self.debug:
                print(f"success: {success}")

                print("--feedback--\n")
                for log in logs:
                    print(log)
                print("\n--")

            # pprint.PrettyPrinter().pprint(resp)

            plan = resp["plan"]
            reason = resp.get("reasoning", "UNDISCLOSED REASONING")

            if self.debug:
                print(f"plan:")
                for action, arg in plan:
                    parsed_arg = arg
                    print(f"\t{action}( {parsed_arg} )")

                print(f"reason: {reason}")

            return resp
        else:
            print(f"[planning-sim] FAILED. resp={resp}, logs={logs}")
            return {}

    def run_planning(
        self,
        *,
        llm_planner: SPINE,
        task: str,
        graph_data_gen: graph_sim.GraphSim,
        max_iterations=10,
    ) -> Dict[str, Any]:
        """Top-level planning loop: query SPINE → execute actions → feed graph diffs back.

        Iterates up to max_iterations times. Each iteration queries the planner for a plan,
        executes the actions on the GraphSim, and feeds any discovered graph updates back
        as the next planner input. Stops early when the planner issues an 'answer' action.
        Returns the final planner response dict.
        """
        done = False
        planner_input = f"task: {task}"

        for _ in range(max_iterations):
            out = self.query_planner(planner_input, llm_planner)

            # if plan is badly formed just return
            if "plan" not in out:
                return {"response": out}

            for step in out["plan"]:
                function, arg = step

                if function == "answer":
                    done = True

                have_updates = graph_data_gen.take_action(function, arg)

                if have_updates:
                    break

            if done:
                break

            planner_input = graph_data_gen.get_updator().form_updates()

        return out
