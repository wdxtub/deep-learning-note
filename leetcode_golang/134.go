// https://leetcode-cn.com/problems/gas-station/

func canCompleteCircuit(gas []int, cost []int) int {
	totalTank, curTank, start := 0, 0, 0
	for i := 0; i < len(gas); i++ {
		totalTank += gas[i] - cost[i]
		curTank += gas[i] - cost[i]
		if curTank < 0 {
			curTank = 0
			start = i + 1
		}
	}
	if totalTank < 0 {
		return -1
	}
	return start
}