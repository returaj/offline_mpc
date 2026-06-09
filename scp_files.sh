read -p "pswd: " pswd

hostname=4a100_5

source_path=/home/cs21d406/github/offline_mpc/dsrl_model/runs/neg_union_full/safecl/pos_1_neg_10_union_500/safemil/lr_0.00001/batch_32_bag_128_hz_50
target_path=/home/returaj/Documents/research/offline_safe_rl/neg_union_full/pos_1_neg_10_union_500/safemil

for task in 'SwimmerVelocity' 'AntVelocity' 'PointCircle2' 'CarCircle2' 'PointGoal1' 'CarGoal1' 'AntCircle' 'AntRun' 'DroneCircle' 'DroneRun'; do
# for task in 'AntCircle' 'AntRun' 'DroneCircle' 'DroneRun'; do
# for task in 'AntVelocity' 'SwimmerVelocity' 'Walker2dVelocity' 'HopperVelocity'; do
    echo "Starting $task"
    for ttype in 'ep_reward' 'ep_cost' 'ep_length' 'ep_worst_cost_0.1' 'ep_worst_cost_0.2' 'ep_worst_cost_0.3' 'ep_worst_cost_0.5' 'dataset'; do
  
       sshpass -p $pswd scp -r $hostname:$source_path$pi/*${task}*/*/seed-00*/*${ttype}_*  $target_path$pi/*${task}*/${ttype}/

       echo "Done ${ttype} for $task" 
    done
done
