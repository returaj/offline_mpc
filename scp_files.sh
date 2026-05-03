read -p "pswd: " pswd

hostname=

source_path=/home/cs21d406/github/offline_mpc/dsrl_model/runs/neg_union_full/safecl/pos_1_neg_5_union_500/safecl/plabel_1.0_nlabel_0.0/limit_0.85_embd_250/batch_128_temp_0.1_hz_100/value_nonorm_scale
target_path=/home/returaj/Documents/research/offline_safe_rl/neg_union_full/pos_1_neg_5_union_500/safecl/value

for task in 'PointCircle2' 'CarCircle2' 'PointGoal1' 'CarGoal1'; do
# for task in 'AntCircle' 'AntRun' 'DroneCircle' 'DroneRun'; do
# for task in 'AntVelocity' 'SwimmerVelocity' 'Walker2dVelocity' 'HopperVelocity'; do
    echo "Starting $task"
    for ttype in 'ep_reward' 'ep_cost' 'ep_length' 'ep_worst_cost_0.1' 'ep_worst_cost_0.2' 'ep_worst_cost_0.3' 'ep_worst_cost_0.5' 'dataset'; do
  
       sshpass -p $pswd scp -r cs21d406@$hostname:$source_path/*${task}*/*/seed-00*/*${ttype}_*  $target_path/*${task}*/${ttype}/

       echo "Done ${ttype} for $task" 
    done
done
