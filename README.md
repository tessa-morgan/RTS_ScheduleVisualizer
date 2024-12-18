# Task Scheduler GUI

This is a Python-based GUI application for simulating and visualizing real-time task scheduling using different algorithms such as Rate Monotonic Scheduling (RMS), Earliest Deadline First (EDF), Least Laxity First (LLF), and Deadline Monotonic Scheduling (DMS). The application helps determine if a set of periodic tasks is schedulable under the selected algorithm and visualizes the schedule.

## Features
- **Supported Algorithms**:
  - Rate Monotonic Scheduling (RMS)
  - Earliest Deadline First (EDF)
  - Least Laxity First (LLF)
  - Deadline Monotonic Scheduling (DMS)
- **Schedulability Analysis**:
  - Checks if tasks are schedulable under the chosen algorithm.
- **Visualization**:
  - Generates and displays a Gantt chart-style visualization of task schedules.
  - Marks task periods with triangles for clarity.

---

## Prerequisites
- Python 3.8+
- Required libraries:
  - `tkinter` (pre-installed with Python)
  - `matplotlib`
  - `numpy`

Install missing libraries using:
```bash
pip install matplotlib numpy
