read -p "pswd: " pswd

hostname=

source_path=/home/cs21d406/github/offline_mpc/dsrl_model/runs/neg_union/safetd3/norm/simultaneous_cost_contrastive/lr_0.00001/batch_128_M_0_hz_5/alpha_0.01_vbeta_1.0_bootlambda_0.0/
target_path=/mnt/d/books/iitm/phd/offline_mpc/safetd3/neg_50/safetd3/simultaneous/hz_5/lambda_0.0/valpha_0.01_vbeta_1.0/contrastive/

# for task in 'Circle2' 'Button1' 'Goal1'; do
for task in 'Ant' 'Swimmer' 'Walker2d'; do
# for task in 'Ant' 'Swimmer' 'Walker2d' 'Circle2' 'Button1' 'Goal1'; do
    echo "Starting $task"
    for ttype in 'ep_reward' 'ep_cost' 'ep_length' 'ep_worst_cost_0.1' 'ep_worst_cost_0.2' 'ep_worst_cost_0.3' 'ep_worst_cost_0.5'; do
        
       sshpass -p $pswd scp -r cs21d406@$hostname:$source_path/*${task}*/*/seed-00*/${ttype}_*  $target_path/*${task}*/${ttype}/

       echo "Done ${ttype} for $task" 
    done
done
