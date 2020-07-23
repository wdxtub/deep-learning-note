// https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/

/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
 func removeNthFromEnd(head *ListNode, n int) *ListNode {
	faker := &ListNode{}
	faker.Next = head
	var pre *ListNode
	cur := faker
	i := 1
	for head != nil {
        if i >= n {
            pre = cur
            cur = cur.Next
        }
        head = head.Next
        i++
	}
	pre.Next = pre.Next.Next
    return faker.Next
}



