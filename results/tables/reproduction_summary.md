| method | scenario | paper_reference_value | reproduced_value_mean | reproduced_value_std | relative_gap_percent | notes |
| --- | --- | --- | --- | --- | --- | --- |
| Static PID | nominal | -1.77 | -1.7692348104058027 | 0.00687329668220137 | 0.04323105051962471 | exact paper table value; evaluated via official constant_gains.npy |
| Pure-RL (paper baseline) | nominal | -2.08 | -6.825496679616239 | 6.151501735232236 | -228.1488788277038 | exact paper table value |
| CIRL reproduced | nominal | -1.33 | -1.602622424958751 | 0.002738935211684822 | -20.497926688627892 | exact paper table value; evaluated via official results_pid_network_rep_newobs_1.pkl -> inter[0]['p_list'][149] |
| Pure-RL (paper baseline) | disturbance | -1.76 | -1.8091384416854972 | 0.5300749281227878 | -2.791956913948704 | exact paper table value |
| CIRL reproduced | disturbance | -1.38 | -1.382205946533039 | 0.013805536823274631 | -0.15985119804630954 | exact paper table value |
| Static PID | highop | -6.81 | -6.928813223865317 | 0.02630579918387419 | -1.744687575114797 | exact paper table value; evaluated via official constant_gains_highop.npy |
| CIRL reproduced | highop | -4.04 | -2.742095861482162 | 0.008830814117311733 | 32.12634006232272 | exact paper table value; evaluated via official low-op CIRL from results_pid_network_rep_newobs_1.pkl -> inter[0]['p_list'][149] |
| CIRL high-op extended (paper reproduction only) | highop | -2.07 | -2.126485252947332 | 0.04731354140334038 | -2.7287561810305467 | exact paper table value |