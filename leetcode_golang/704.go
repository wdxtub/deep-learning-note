// https://leetcode-cn.com/problems/binary-search/

func search(nums []int, target int) int {
	if len(nums) == 0 {
		return -1
	}
    if len(nums) == 1 {
        if nums[0] == target {
            return 0
        }
        return -1
    }
	start, end := 0, len(nums) - 1
	for start < end {
		mid := (start + end) / 2
		if nums[mid] == target{
			return mid
		}
        if nums[start] == target {
            return start 
        }
        if nums[end] == target {
            return end
        }
        if end - start == 1 {
            return -1 
        }
		if nums[mid] > target {
			end = mid
		} else {
			start = mid
		}
	}
	return -1
}