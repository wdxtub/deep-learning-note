// https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/

func removeDuplicates(nums []int) int {
	count := 0
	for i := 0; i < len(nums); i++ {
		if i == 0 {
			nums[count] = nums[i] 
			count += 1
		} else {
			if nums[i] > nums[count-1] {
				nums[count] = nums[i]
				count += 1
			}
		}
	}
	return count
}