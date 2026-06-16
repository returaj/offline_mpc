read -p "pswd: " pswd

hostname=helix_3

source_path=/home/cs21d406/github/offline_mpc/dsrl_model/runs/neg_union_full/safecl/pos_1_neg_10_union_500/safecl/plabel_1.0_nlabel_-1.0/limit_0.90_embd_1000/batch_64_vtemp_0.2_pitemp_0.1_hz_50/norm_unionweight0.99_piq0.9
target_path=/home/returaj/Documents/research/offline_safe_rl/neg_union_full/pos_1_neg_10_union_500/safecl/dual_sink_b64_attnunion_uniondecay_piq/piq_temp_0.1_baseline_0.9

for task in 'AntVelocity' 'PointCircle2' 'AntRun' 'DroneRun' 'SwimmerVelocity' 'CarCircle2' 'PointGoal1' 'CarGoal1' 'AntCircle' 'DroneCircle'; do
# for task in 'AntCircle' 'AntRun' 'DroneCircle' 'DroneRun'; do
# for task in 'AntVelocity' 'SwimmerVelocity' 'Walker2dVelocity' 'HopperVelocity'; do
    echo "Starting $task"
    for ttype in 'ep_reward' 'ep_cost' 'ep_length' 'ep_worst_cost_0.1' 'ep_worst_cost_0.2' 'ep_worst_cost_0.3' 'ep_worst_cost_0.5' 'dataset'; do
  
       sshpass -p $pswd scp -r $hostname:$source_path$pi/*${task}*/*/seed-00*/*${ttype}_*  $target_path$pi/*${task}*/${ttype}/

       echo "Done ${ttype} for $task" 
    done
done
