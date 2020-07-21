// https://leetcode-cn.com/problems/merge-sorted-array/

func merge(nums1 []int, m int, nums2 []int, n int)  {
	p1 := m - 1
	p2 := n - 1
	p := m + n -1
	for {
		if p1 >=0 && p2 >= 0 {
			if nums1[p1] < nums2[p2] {
				nums1[p--] = nums2[p2--]
			} else {
				nums1[p--] = nums1[p1--]
			}
		} else {
			break
		}
	}
	for i := p2; i >= 0; i-- {
		nums1[p--] = nums[p2--]
	}
}