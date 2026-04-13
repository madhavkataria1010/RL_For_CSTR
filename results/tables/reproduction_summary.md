| method | scenario | paper_reference_value | reproduced_value_mean | reproduced_value_std | relative_gap_percent | notes |
| --- | --- | --- | --- | --- | --- | --- |
| Static PID | nominal | -1.77 | -1.7734895045012264 | 0.005131327018575401 | -0.19714714696194496 | exact paper table value; evaluated via official constant_gains.npy |
| Pure-RL (paper baseline) | nominal | -2.08 | nan | nan | nan | Error(s) in loading state_dict for PureRLPaperPolicy:
	Missing key(s) in state_dict: "output.weight", "output.bias". 
	Unexpected key(s) in state_dict: "output_mu.weight", "output_mu.bias", "output_std.weight", "output_std.bias". ; evaluated via official results_rl_network_rep_newobs_0.pkl -> inter[1]['p_list'][149] |
| CIRL reproduced | nominal | -1.33 | nan | nan | nan | Error(s) in loading state_dict for CIRLPolicy:
	Missing key(s) in state_dict: "output.weight", "output.bias". 
	Unexpected key(s) in state_dict: "output_mu.weight", "output_mu.bias", "output_std.weight", "output_std.bias". ; evaluated via official results_pid_network_rep_newobs_1.pkl -> inter[0]['p_list'][149] |
| Pure-RL (paper baseline) | disturbance | -1.76 | -1184.201790567027 | 3.423251724476817 | -67184.19264585381 | exact paper table value; evaluated via official best_policy_rl_dist.pth |
| CIRL reproduced | disturbance | -1.38 | -2.2404133744516153 | 0.013497093664430823 | -62.34879525011706 | exact paper table value; evaluated via official best_policy_pid_dist_0.pth |
| Static PID | highop | -6.81 | -6.947269740959623 | 0.010284631831605294 | -2.0157083841354377 | exact paper table value; evaluated via official constant_gains_highop.npy |
| CIRL reproduced | highop | -4.04 | nan | nan | nan | Error(s) in loading state_dict for CIRLPolicy:
	Missing key(s) in state_dict: "output.weight", "output.bias". 
	Unexpected key(s) in state_dict: "output_mu.weight", "output_mu.bias", "output_std.weight", "output_std.bias". ; evaluated via official low-op CIRL from results_pid_network_rep_newobs_1.pkl -> inter[0]['p_list'][149] |
| CIRL high-op extended (paper reproduction only) | highop | -2.07 | -9.120706020736675 | 0.010903617560067438 | -340.6138174268925 | exact paper table value; evaluated via official best_policy_pid_highop_0.pth |