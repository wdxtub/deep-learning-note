// https://leetcode-cn.com/problems/maximum-subarray/

func maxSubArray(nums []int) int {
	if len(nums) == 0 {
		return 0
	}
	max := nums[0]
	sofar := nums[0]
	for i := 1; i < len(nums); i++ {
		if sofar + nums[i] > nums[i] {
			sofar = sofar + nums[i]
		} else {
			sofar = nums[i]
		}
		if sofar > max {
			max = sofar
		}
	}
	return max
}