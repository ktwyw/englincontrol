# Course notebooks

Twenty executed notebooks on linear control theory, built on one working method: model the plant, check what
the model implies (poles, response, frequency response, limits), design, and verify on the loop that is
actually closed - on data where data exist. Every notebook states its learning objectives, prerequisites and
study time, rebuilds a key algorithm in plain Python ("Inside the algorithm", checked against the library),
and ends with exercises including an "Implement it yourself" task. Worked solutions are in
[`../solutions`](../solutions).

| # | Notebook | Time | Before you start | Solutions |
|---|---|---|---|---|
| [00](00_models_of_linear_systems.ipynb) | Models of linear systems | 60 min | linear ODEs, the Laplace transform, basic linear algebra | [solutions](../solutions/00_solutions.ipynb) |
| [01](01_time_response.ipynb) | Time response | 60 min | notebook 00 | [solutions](../solutions/01_solutions.ipynb) |
| [02](02_frequency_response.ipynb) | Frequency response | 75 min | notebooks 00 and 01 | [solutions](../solutions/02_solutions.ipynb) |
| [03](03_stability_tests.ipynb) | Stability tests | 60 min | notebook 00 | [solutions](../solutions/03_solutions.ipynb) |
| [04](04_the_root_locus.ipynb) | The root locus | 75 min | notebooks 01 and 03 | [solutions](../solutions/04_solutions.ipynb) |
| [05](05_the_nyquist_criterion.ipynb) | The Nyquist criterion | 75 min | notebooks 02 and 03 | [solutions](../solutions/05_solutions.ipynb) |
| [06](06_pid_control_and_tuning.ipynb) | PID control and tuning | 90 min | notebooks 01 and 02 | [solutions](../solutions/06_solutions.ipynb) |
| [07](07_lead_lag_and_loop_shaping.ipynb) | Lead, lag and loop shaping | 75 min | notebooks 02 and 06 | [solutions](../solutions/07_solutions.ipynb) |
| [08](08_controllability_observability_and_pole_placement.ipynb) | Controllability, observability and pole placement | 75 min | notebooks 00 and 01 | [solutions](../solutions/08_solutions.ipynb) |
| [09](09_observers_and_the_separation_principle.ipynb) | Observers and the separation principle | 60 min | notebook 08 | [solutions](../solutions/09_solutions.ipynb) |
| [10](10_the_linear_quadratic_regulator.ipynb) | The linear-quadratic regulator | 75 min | notebook 08 | [solutions](../solutions/10_solutions.ipynb) |
| [11](11_kalman_filtering_and_lqg.ipynb) | Kalman filtering and LQG | 90 min | notebooks 09 and 10 | [solutions](../solutions/11_solutions.ipynb) |
| [12](12_sampling_and_discretisation.ipynb) | Sampling and discretisation | 60 min | notebook 01 | [solutions](../solutions/12_solutions.ipynb) |
| [13](13_digital_control_design.ipynb) | Digital control design | 90 min | notebooks 07, 08, 10 and 12 | [solutions](../solutions/13_solutions.ipynb) |
| [14](14_robustness_and_limitations.ipynb) | Robustness and fundamental limitations | 75 min | notebooks 02, 05 and 07 | [solutions](../solutions/14_solutions.ipynb) |
| [15](15_system_identification.ipynb) | System identification | 90 min | notebooks 01, 02 and 12 | [solutions](../solutions/15_solutions.ipynb) |
| [16](16_model_reduction_and_realisation.ipynb) | Model reduction and realisation theory | 60 min | notebooks 00 and 08 | [solutions](../solutions/16_solutions.ipynb) |
| [17](17_multivariable_basics.ipynb) | Multivariable basics | 75 min | notebooks 06 and 10 | [solutions](../solutions/17_solutions.ipynb) |
| [18](18_numerics_of_control.ipynb) | The numerics of control | 75 min | notebooks 03, 10 and 16 | [solutions](../solutions/18_solutions.ipynb) |
| [19](19_project_a_position_servo_from_data.ipynb) | Project: a position servo from measured data to a digital controller | 4 h | the whole course | [solutions](../solutions/19_solutions.ipynb) |

**Suggested order:** foundations (00-05: models, responses, stability, root locus, Nyquist), classical design
(06-07), state-space design (08-11), digital control (12-13), robustness (14), identification and reduction
(15-16), multivariable and numerics (17-18), and the project (19), which draws on all of them.

The notebooks are generated from `build_notebooks.py` (`python notebooks/build_notebooks.py` rebuilds and executes
them; pass notebook numbers to rebuild only some). They open in Google Colab, where the first cell installs
`englincontrol` from GitHub.
