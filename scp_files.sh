read -p "pswd: " pswd

hostname=andromeda_2

source_path=/home/cs21d406/github/offline_mpc/dsrl_model/runs/neg_union_full/safecl/pos_1_neg_0_union_500/safecl_norandomtraj/plabel_1.0_nlabel_-1.0/limit_0.90_embdsize_64_embdfreq_1000/batch_64_vtemp_1.0_constant_pitemp_0.2_pialpha_0.9_hz_50/weightdecay_true/norm_piq_starttraj_removeunionlmbda_constant_projection/
target_path=/home/returaj/Documents/research/offline_safe_rl/neg_union_full/pos_1_neg_0_union_500/safecl_il_noreg

for task in 'AntVelocity' 'PointCircle2' 'AntRun' 'DroneRun' 'SwimmerVelocity' 'CarCircle2' 'AntCircle' 'DroneCircle' 'PointGoal1' 'CarGoal1'; do
# for task in 'AntCircle' 'DroneCircle' 'CarCircle2'; do
# for task in 'AntVelocity' 'SwimmerVelocity' 'Walker2dVelocity' 'HopperVelocity'; do
    echo "Starting $task"
    for ttype in 'ep_reward' 'ep_cost' 'ep_length' 'ep_worst_cost_0.1' 'ep_worst_cost_0.2' 'ep_worst_cost_0.3' 'ep_worst_cost_0.5' 'dataset'; do
  
       sshpass -p $pswd scp -r $hostname:$source_path$pi/*${task}*/*/seed-00*/*${ttype}_*  $target_path$pi/*${task}*/${ttype}/

       echo "Done ${ttype} for $task" 
    done
done
