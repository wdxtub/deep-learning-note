// https://leetcode-cn.com/problems/add-two-numbers/

/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	result := ListNode{
		Val: 0,
		Next: nil,
	}
	current := &result
	prev_padding := 0
	for {
		value := 0
		value += prev_padding
		if l1 != nil {
			value += l1.Val
			l1 = l1.Next
		}
		if l2 != nil {
			value += l2.Val
			l2 = l2.Next
		}
		if value > 9 {
			value -= 10
			prev_padding = 1	
		} else {
			prev_padding = 0
		}	

		current.Val = value
		if l1 == nil && l2 == nil {
			if prev_padding > 0 {
				current.Next = &ListNode{
					Val: prev_padding,
					Next: nil,
				}
			}
			break
		}

		current.Next = &ListNode{
			Val: 0,
			Next: nil,
		}
		current = current.Next
	}
	return &result
}