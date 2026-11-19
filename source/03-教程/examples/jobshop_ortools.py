"""精简改编 Google OR-Tools 官方最小 Job Shop 示例。"""

from __future__ import annotations

from collections import defaultdict

from ortools.sat.python import cp_model


JOBS_DATA = [
    [(0, 3), (1, 2), (2, 2)],
    [(0, 2), (2, 1), (1, 4)],
    [(1, 4), (2, 3)],
]


def main() -> None:
    """建立 Job Shop 模型，打印每台机器的任务区间。"""
    machine_count = 1 + max(machine for job in JOBS_DATA for machine, _ in job)
    horizon = sum(duration for job in JOBS_DATA for _, duration in job)
    model = cp_model.CpModel()
    machine_intervals: dict[int, list[cp_model.IntervalVar]] = defaultdict(list)
    task_vars: dict[tuple[int, int], tuple[cp_model.IntVar, cp_model.IntVar]] = {}

    for job_id, job in enumerate(JOBS_DATA):
        previous_end = None
        for task_id, (machine_id, duration) in enumerate(job):
            start = model.new_int_var(0, horizon, f"start_{job_id}_{task_id}")
            end = model.new_int_var(0, horizon, f"end_{job_id}_{task_id}")
            interval = model.new_interval_var(
                start, duration, end, f"interval_{job_id}_{task_id}"
            )
            if previous_end is not None:
                # 同一作业的下一道工序必须等上一道完成。
                model.add(start >= previous_end)
            previous_end = end
            machine_intervals[machine_id].append(interval)
            task_vars[job_id, task_id] = (start, end)

    for machine_id in range(machine_count):
        # 同一机器在同一时间只能处理一道工序。
        model.add_no_overlap(machine_intervals[machine_id])

    makespan = model.new_int_var(0, horizon, "makespan")
    final_ends = [task_vars[job_id, len(job) - 1][1] for job_id, job in enumerate(JOBS_DATA)]
    model.add_max_equality(makespan, final_ends)
    model.minimize(makespan)

    solver = cp_model.CpSolver()
    status = solver.solve(model)
    print(f"status={solver.status_name(status)}")
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("No solution found")
        return

    print(f"Optimal Schedule Length: {solver.value(makespan)}")
    for machine_id in range(machine_count):
        assigned = []
        for job_id, job in enumerate(JOBS_DATA):
            for task_id, (machine, duration) in enumerate(job):
                if machine != machine_id:
                    continue
                start, end = task_vars[job_id, task_id]
                assigned.append(
                    (solver.value(start), solver.value(end), job_id, task_id, duration)
                )
        assigned.sort()
        rendered = ", ".join(
            f"J{job_id}-T{task_id}[{start},{end})"
            for start, end, job_id, task_id, _ in assigned
        )
        print(f"Machine {machine_id}: {rendered}")


if __name__ == "__main__":
    main()
