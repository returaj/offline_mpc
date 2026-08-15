read -p "pswd: " pswd

hostname=eyeofgod_4

source_path=/home/cs21d406/github/offline_mpc/dsrl_model/runs/neg_union_full/safecl/pos_1_neg_10_union_500/safecl_weight_based/plabel_1.0_nlabel_-1.0/lmbda_0.8_limit_0.90_embdsize_64_embdfreq_1000/batch_64_constant_pitemp_*_pialpha_*_hz_50/weightdecay_true/norm_constant_projection/
target_path=/home/returaj/Documents/research/offline_safe_rl/neg_union_full/pos_1_neg_10_union_500/num_comp_1/safecl/embd_64_piconst_0.9_wdecay_fulltrans/ablation/lmbda/lmbda_0.8/

# for task in 'AntVelocity' 'PointCircle2' 'AntRun' 'DroneRun' 'SwimmerVelocity' 'AntCircle' 'DroneCircle' 'PointGoal1' 'CarGoal1' 'CarCircle2' ; do
for task in 'DroneRun' 'PointCircle2'; do
# for task in 'AntCircle' 'DroneCircle' 'CarCircle2'; do
# for task in 'AntVelocity' 'SwimmerVelocity' 'Walker2dVelocity' 'HopperVelocity'; do
    echo "Starting $task"
    for ttype in 'ep_reward' 'ep_cost' 'ep_length' 'ep_worst_cost_0.1' 'ep_worst_cost_0.2' 'ep_worst_cost_0.3' 'ep_worst_cost_0.5' 'dataset' 'progress'; do
  
       sshpass -p $pswd scp -r $hostname:$source_path/*${task}*/*/seed-00*/*${ttype}_*  $target_path/*${task}*/${ttype}*/

       echo "Done ${ttype} for $task" 
    done
done
