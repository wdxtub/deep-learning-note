// https://leetcode-cn.com/problems/intersection-of-two-linked-lists/

/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func getIntersectionNode(headA, headB *ListNode) *ListNode {
	p1 := headA
	p2 := headB
	p1End := false
	p2End := false
	if headA == nil || headB == nil {
		return nil
	}
	
	for {
		if p1 == p2 {
			return p1
		}
		if p1.Next != nil {
			p1 = p1.Next
		} else {
			p1 = headB
			if p1End {
				return nil
			}
			p1End = true
		}

		if p2.Next != nil {
			p2 = p2.Next
		} else {
			p2 = headA
			if p2End {
				return nil
			}
			p2End = true
		}
	}
}