DT_DIR=`pwd`
TASK=$1

for((time=3;time<4;time++))
do
	for CELLTYPE in 'GRU'
	do
		echo "***************** TASK-$TASK CELLTYPE-$CELLTYPE Time-$time Train******************"
		echo `date`
		python Test.py $TASK $CELLTYPE $TASK-$CELLTYPE-$time
		echo `date`
	done
done
