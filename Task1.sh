DT_DIR=`pwd`
TASK=$1

for((time=2;time<3;time++))
do
	for CELLTYPE in 'GRU'
	do
		echo "***************** TASK-$TASK CELLTYPE-$CELLTYPE Time-$time Train******************"
		echo `date`
		python Train_Core1.py $TASK $CELLTYPE $TASK-$CELLTYPE-$time
		echo `date`
	done
done
