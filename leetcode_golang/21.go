// https://leetcode-cn.com/problems/merge-two-sorted-lists/

/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
 func mergeTwoLists(l1 *ListNode, l2 *ListNode) *ListNode {
	var head *ListNode
	var cur *ListNode
    if l1 == nil {
        return l2
    }
    if l2 == nil {
        return l1
    }
	if l1.Val <= l2.Val {
		head = l1
		cur = head
		l1 = l1.Next
	} else {
        head = l2
        cur = head
        l2 = l2.Next
    }
	for {
		if l1 == nil && l2 == nil {
			break
		}
		if l1 == nil {
			cur.Next = l2
			break
		}
		if l2 == nil {
			cur.Next = l1
			break
		}
		if l1.Val <= l2.Val {
			cur.Next = l1
			l1 = l1.Next
		} else {
			cur.Next = l2
			l2 = l2.Next
		}
		cur = cur.Next
		
	}
	return head
}