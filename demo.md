
# Elements of Programming Interviews
[TOC]

#Primitive Types

####**Problem 5.4**

Suppose $x\in S_{k}$ if $k$ bits in $x$ are set to 1. How would you compute $y\in S_{k}$  such that $\left|y-x\right|$  is minimal?

**Solution**

Iterate through the bit representation of x from least significant bit to most significant bit and swap the first pair of bits that differ. Intuitively this works because we want to change the fewest significant bits possible. Note that simply swapping the least significant bit with the next least significant bit won't work: ```1011100``` is a counterexample.

####**Problem 5.5**
Compute the [powerset](https://en.wikipedia.org/wiki/Power_set) of a set $S$.

**Solution**
There are two ways to do this: create a bitstring where the index of the bit represents inclusion or exclusion then iterate through all bitstrings by iterating through their decimal representation or use recursion. Bitstring solution has running time $O\left(2^{\left|S\right|}\right)$. The recursive solution has running time $O\left(\left|S\right|2^{\left|S\right|}\right)$ and looks like

```python
def powerset(lst):
    if not lst:
        return [[]]
    without_first = powerset(lst[1:])
    with_first = [ [lst[0]] + rest for rest in without_first ]
    return with_first + without_first
```

####**Variant 5.5.1**
Compute all subsets of size $k$ of a set $S$.

**Solution**
For this variant you need to use a different form the powerset function

```python
def powerset(lst,crnt,i,k):
    if i == len(lst):
        return [crnt] 
    with_next = powerset(lst,crnt+[lst[i]],i+1,k)
    without_next = powerset(lst,crnt,i+1,k)
    return with_next + without_next
```

then the solution is a simple modification

```python
def powerset(lst,crnt,i,k):
    if len(crnt) == k:
        return [crnt]
    if i == len(lst):
        return []
    with_next = powerset(lst,crnt+[lst[i]],i+1,k)
    without_next = powerset(lst,crnt,i+1,k)
    return with_next + without_next
```

####**Variant 5.10.1**
[Euclid's extended algorithm](https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm) for GCD of $a$ and $b$, including [Bézout's coefficients](https://en.wikipedia.org/wiki/B%C3%A9zout%27s_identity) .


**Solution**
Two solutions; the recursive one is easy. Suppose $d = ax+by$ and $a>b$, i.e. 

$$ \gcd\left(a,b\right) = ax+by$$

then it's also the case that  
$$
\begin{align}
\gcd\left(a,b\right)& = bx'+\left(a~\text{mod}~b\right)y'\\
			     & = bx' + \left(a - \left\lfloor \frac{a}{b} \right\rfloor \cdot b \right)y' \\
			     & = ay' + \left(x - \left\lfloor \frac{a}{b} \right\rfloor \cdot y' \right) b
\end{align}
$$

Comparing coefficients of $a$ and $b$ we get that 
$$
\begin{align}
x & = y'\\
y &= \left(x - \left\lfloor \frac{a}{b} \right\rfloor \cdot y' \right) 
\end{align}
$$

```python
def euclids_aglo_rec(a,b):
    assert a > b	
    if b == 0:
        return a,1,0
    else:
    	q = a//b
        d,xp,yp = euclids_aglo_rec(b,a%b)
        x,y = yp, xp-q*yp
        return d,x,y

```

The iterative one is trickier to prove but follows essentially the same form

```python
def euclids_aglo_iter(a,b):
    assert a > b
    x0,y0,x1,y1 = 1,0,0,1
    while b !=0:
        q,a,b = a//b,b,a%b
        x0,x1 = x1,x0-q*x1
        y0,y1 = y1,y0-q*y1
    return a,x0,y0

```

####**Probem 5.11**
For $n \geq 2$ compute all of the primes between $1$ and $n$.

**Solution**
Use the [Sieve of Erasthones](https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes) with some optimizations.

1. We don't need to check even numbers.
2. If $m$ is a multiple of $i$ and $i$ is a multiple of a prime $p$, then it's already eliminated from contention, so we don't check it.
3. If $m$ is not prime, then it has a factor $\leq \sqrt{m}$. Thus we do not need to check multiples of $i$ smaller than $i^2$.
4. A corollary of (3) is that we don't need to look for primes greater than $\sqrt{n}+1$.

```python
def sieve(n):
    nums = [True if i%2==1 else False for i in range(n)]
    for i in range(3,sqrt(n)+1,2):
    	if not nums[i]:
    		continue
	else:
		for j in range(i**2,n,i):
			nums[j] = False
    return [i for i,v in enumerate(nums) if v]
```

####**Variant 5.12.2**
Check if two rectanges, not necessarily $xy$-aligned intersect.

**Solution**
Check $4 \times 4 = 16$ intersections of all edges. Line segment intersection checking is easy (cross-products) but there's an edge case: one point intersection.

```python
def vec_diff(p1,p2):
	return p1[0]-p2[0],p1[1]-p2[1]

def cross(v1,v2):
	# if v1 is counter-clockwise of v2 then result if positive
	x1,y1 = v1
	x2,y2 = v2
	return x1*y2 - y1*x2

def on_seg(pi,pj,pk)
	# pk is the point being tested for colinearity with the line segment pipj
	# this is not in general; only valid when (pj-pi) x (pk-pi) = 0
	(xi,yi) = pi
	(xj,yj) = pj
	(xk,yk) = pk
	return min(xi,xj) <= xk <= max(xi,xj) and min(yi,yj) <= yk <= max(yi,yj)

def intersect(l1,l2):
    # l = ((x1,y1),(x2,y2))
    p1,p2 = l1
    p3,p4 = l2
    # test whether start point of l1 is on the left/right of l2
    d1 = cross(vec_diff(p1,p3),vec_diff(p4,p3)) 
    # test whether end point of l1 is on the left/right of l2
    d2 = cross(vec_diff(p2,p3),vec_diff(p4,p3))	
    # test whether start point of l2 is on the left/right of l1    
    d3 = cross(vec_diff(p3,p1),vec_diff(p2,p1))	    
    # test whether end point of l2 is on the left/right of l1    
    d4 = cross(vec_diff(p4,p1),vec_diff(p2,p1))	
    if d1*d2 < 0 or d3*d4 < 0:
    	return True
    # either there's a one point intersection somewhere (in which case one 
    # will return True) or they don't intersect and so all of these will be false
    return (d1 == 0 and on_seg(p3,p4,p1) or (d2 == 0 and on_seg(p3,p4,p2) or \
	   (d3 == 0 and on_seg(p1,p2,p3) or (d2 == 0 and on_seg(p1,p2,p4)
```

#Arrays and Strings

####**Problem 6.1**
Write a function that partitions an array `A` around a given element `A[i]` into three sets: $\leq, =, >$.

**Solution**
This is similar to the partition function for quicksort.

```python
def partition(A,i):
	pivot = A[i]
	# maintain the follow invariants
	# bottom group: A[0:smaller]
	# middle group: A[smaller:equal]
	# unclassified group: A[equal:larger]
	# top: A[larger:] 
	smaller = equal = 0
	larger = len(A) - 1
	while equal <= larger:
		if A[equal] < pivot:
			A[smaller], A[equal] = A[equal], A[smaller]
			smaller +=1
			equal += 1
		elif A[equal] == pivot:
			equal += 1
		else: # A[equal] > pivot
			A[equal], A[larger] = A[larger], A[equal]
			larger -= 1
	return smaller, larger # smaller is the beginning of middle, 
	# and larger is beginning of top
```

####**Variant 6.1.1**
Assuming keys take one of three values, reorder the array so as to group.

**Solution**
Well if they're integers then just use the same `partition(A,i)` for arbitrary `i`. If they're not then write a comparator for them. Call this function `partition_three(A)`.

####**Variant 6.1.2**
Assuming keys take one of four values, reorder the array so as to group.

**Solution**

Run `partition_three(A)` twice, but on the second run use `A[smaller:]` or `A[:smaller]` depending on whether `top` or `bottom` subarrays end up being homogeneous.


####**Variant 6.1.3**
Assuming keys are Boolean, reorder the array so as to group.

**Solution**

Just impose the ordering `False < True` and use `partition(A,i)`. Python implicitly already imposes this ordering so it already works (in Python). One of `top` or `bottom` subarrays will be empty but it doesn't matter. 
        
####**Problem 6.2**
Design a deterministic scheme by which reads and writes to an unitialized array can be made in $O\left(1\right)$ time. You may use $O\left(n\right)$ additional storage; reads to uninitialized entries should return `False`.

**Solution**

This is exactly like problem 11.1-4 in CLRS so first let me recapitulate that problem:

>We wish to implement a dictionary by using direct addressing on a huge array. At the start, the array entries may contain garbage, and initializing the entire array is impractical because of its size. Describe a scheme for implementing a direct-address dictionary on a huge array. Each stored object should use O (1) space; the operations ```insert```, ```delete```, and ```search``` should take $O\left(1\right)$ time each; and initializing the data structure should take $O\left(1\right)$ time.

The solution there is to use a stack to store the actual values. Let `huge` be the array and `stack` be the stack. To **insert** an element into `huge` with key `x`, append `x` to the stack and store at `huge[x]`. the length of the stack, i.e. the index of `x` in the stack array. To **search** for an entry `y` first check that `huge[y] < len(stack)` and that `stack[huge[y]] == y`. To **delete** an element `x` swap the top of the stack with the element to be deleted, update the relevant entries in `huge`, and then pop the stack:
```python
stack[huge[x]] = stack[-1]
huge[stack[-1]] = huge[x]
huge[x] = None
stack.pop()
```

For this problem the solution is exactly the same, with searching to be understood as reading.

####**Problem 6.3**
Design an algorithm that takes a sequence of $n$ three-dimensional coordinates to be traversed, and returns the minimum battery capacity needed to complete the journey (assuming go up discharges and go down recharges).

**Solution**

This is similar to the stock profit problem (find the lowest stock price and highest stock price, in order to profit the most). The solution to that problem is to iterate through array and keep track of the minimum seen so far, and the maximum profit possible (by taking the difference between the minimum seen so far and the current entry).

The solution here is to do the same thing but just discard all of entries in the coordinates except the height. That's largest battery capacity you need.

```python
def battery(A):
	heights = [z for x,y,z in A]
	hmin = heights[0]
	capacity = 0
	for h in heights[1:]:
		capacity = max(capacity, h - hmin)
		hmin = min(h,minh)
	return capcity
```	

####**Problem 6.4**
For array `A` compute each of the following

1. $\max_{i_0<j_0<i_1<j_1} \big(A\left[j_0\right] - A\left[i_0\right]) + A\left[j_1\right] - A\left[i_1\right])\big)$.
2. With $i_0 < j_0 < i_1 < j_1 < \cdots < i_{k-1} < j_{k-1}$, for at most $k$ terms, maximize
$$ \sum_{t=0}^{k-1} A\left[j_t\right] - A\left[i_t\right] $$
3. (2) but with exactly $k$ terms.
4. (2) but now with $k$ can be any value from $0$ to $\lfloor n/2 \rfloor$.


**Solution**

1. Suppose we were just maximizing $\big(A\left[j_0\right] - A\left[i_0\right])\big)$. How would you do that? well that's just the stock profit problem (or the immediately preceding problem). In order to use this just partition `A` on $j$, i.e. into `A[0:j]` and `A[j:]` and run the "stock price" algorithm on both portions. Repeat this for each $j$ and you get an $O\left(n^2\right)$ algorithm. 

		def max_pair_diffs(A):
			pair_max = 0
			for i in range(2,len(A)-2+1):
				pair_max = max(pair_max, battery(A[:i])+battery(A[i:])
			return pair_max

	But you can actually do better: run the forward iteration storing the results in an array then, run a reverse iteration (flipping what you're looking for) and combine the results with the stored results from the forward iteration.

		def max_pair_diffs(A):
		    forward_results = []
		    amin = A[0]
		    total = 0
		    for a in A[1:-1]:
				total = max(capacity, a - amin)
				amin = min(a,amin)
				forward_results.append(total)
		    amax = A[-1]
		    total = 0
		    pair_max = 0
		    for i in range(len(A)-2,1,-1):
				a = A[i]
				total = max(capacity, amax - a)
				amax = max(a,amax)
				pair_max = max(pair_max,forward_results[i-1]+total)
		    return pair_max
		    
2. There are two ways to solve this. The simpler solution is similar to the backwards iteration of `max_pair_diffs`:

		def kdiffs(A):
		    maxsf = 0
		    total = 0
		    for a in reversed(arr):
		    	if a<maxsf:
	        	    total += maxsf-a
			else:
			    maxsf = a
		    return total

	Running time is $O\left(n\right)$. The more difficult solution is DP. With $0\leq i\leq k$ terms, using upto, and including, `A[j]`:
$$total\left[i\right]\left[j\right] = \max \begin{cases}
                                        total\left[i\right]\left[j-1\right] & \text{don't include } A\left[j\right] \\
                                        \max_{0\leq m \leq j-1} \left\{A\left[j\right] - A\left[m\right] + total\left[i-1\right]\left[m\right] \right\} & \text{include } A\left[j\right] 
                                        \end{cases}$$                                      
with base cases $total\left[i\right]\left[0\right] = 0$ (includes only 1 term) and $total\left[0\right]\left[j\right]$ (no pairs). This is $O\left(kn^2\right)$.

	There's a further optimization that gets the complexity of the DP solution down to $O\left(kn\right)$: notice that 
	$$ \begin{align}
		\max_{0\leq m \leq j-1} \left\{A\left[j\right] - A\left[m\right] + total\left[i-1\right]\left[m\right] \right\} &= A\left[j\right] +   \max_{0\leq m \leq j-1}  \left\{total\left[i-1\right]\left[m\right] - A\left[m\right]  \right\} \\
		&= A\left[j\right] +   \max  \left\{D, total\left[i-1\right]\left[j-1\right] - A\left[j-1\right]  \right\} 
		\end{align}$$
	where $D = \max_{0\leq m \leq j-2}  \left\{total\left[i-1\right]\left[m\right] - A\left[m\right]  \right\}$. Therefore keeping track of $D$ enables us to update the inner $\max$ in constant time.
3. For exactly $k$ terms simply remove the "don't include" branch in the DP solution.
4. Surprisingly this is trivial: just pick all pairs $A\left[i\right],A\left[i+1\right]$ such that $A\left[i\right]<A\left[i+1\right]$.


####**Problem 6.5**
Design an efficient algorithm for the $0 \text{ mod } n$-sum subset problem.

**Solution** 

Let `A` be an array of length $n$. First a little theory: $x \text{ mod } n$ can only take on values $0,1,\dots,n-1$. Therefore, for example
$$ \sum_{i=0}^{n-1}A\left[i\right] \text{ mod } n \in \left\{0,1,\dots,n-1\right\} $$
Hence let
$$prefix\left(j\right) := \sum_{i=0}^{j-1}A\left[i\right] \text{ mod } n $$

Either for some $j$ it's the case that $$prefix\left(j\right)$$ equals 0, or $$prefix\left(j\right)=prefix\left(j'\right)$$ for some $j,j'$, in which case their difference equals zero. But since $$prefix\left(j\right) $$ is a prefix sum 

$$ prefix\left(j\right)- prefix\left(j'\right)  = \sum_{i=j'+1}^j A\left[i\right] \text{ mod } n$$
and therefore `A[j'+1:j+1]` is the subarray whose sum equals $0 \text{ mod } n$.

```python
def nsubset_sum(A):
    n = len(A)
    prefix_sum = []
    buckets = [None for i in range(n)] 
    for i in range(n):
        prefix_sum.append(sum(A[:i+1])%n)
        if prefix_sum[-1] == 0:
            return A[:i+1]
        if not buckets[prefix_sum[-1]]:
            buckets[prefix_sum[-1]] = i
        else:
            return A[buckets[prefix_sum[-1]]+1:i+1] 
```            

####**Problem 6.6**

Design an efficient algorithm that computes longest increasing subarray. 

**Solution** 

The "brute-force" algorithm is to compute for each index $i$ the length $m_i$ of the longest increasing subarray ending at $i$. This is a sort of DP algorith, where you extend the subarray ending at $A\left[i-1\right]$ iff $A\left[i-1\right] < A\left[i\right]$. 

```python
    def increasing_subarray(A):
        lmax = 1
        gmax = 1
        for i in range(len(A)):
            if A[i] < A[i+1]:
                lmax += 1
                gmax = max(gmax,lmax)
            else:
                lmax = 1
        return gmax
```            

####**Variant 6.8**

Offline maximum

**Solution** 

This is problem 21-1 from CLRS, so I'll quote it

>We want to determine the $m$th minimum that would be extracted from a set of numbers with maximum $n$, given a particular sequence of insertions and extractions.

The solution is to use the Union-Find data structure

```python
def find_set(u, parent):
    if u != parent[u]:
        parent[u] = find_set(parent[u], parent)
    return parent[u]

def union(i,j, parent, rank):
    a,b = find_set(i, parent), find_set(j, parent)
    if a != b:
        if rank[a] > rank[b]:
            parent[b] = a
            rank[a] += 1
        elif rank[b] > rank[a]:
            parent[a] = b
            rank[b] += 1
        else: # rank[a] = rank[b]
            parent[b] = a
            rank[a] += 1
```

Then partition the sequence of operations in $E_j$, the $j$th extraction, and $I_j$, all of the insertions between the $j-1$th and $j$th extraction. Note $I_j$ might be empty (if there are consecutive extractions). Make each $I_j$ a set (in the Union-Find sense). The algorithm is tedious and needs to consider the edge case of consecutive extractions in particular.

```python
import queue
def offline_min(A,ops,n):
    # 2*n so that there are dummy sets
    parent = [i for i in range(2*n)]
    rank = [1 for i in range(2*n)]
    # ops is the sequence of operations: 0 for insert, 1 for extract
    assert ops[0] == 0 # can't extract before any insertions
    A = queue.deque(A) # so that we can popleft
    # each set corresponds to a group of insertions
    # associate with the root of that the order of that group
    # i.e. first group of insertions, second, etc.
    root_pos = {}
    # in order to keep track of the positions of all of the sets
    # so that we can "lump forward"
    pos_root = []
    num_ex = 0
    i = A[0] # create first group of insertions
    for j,op in enumerate(ops):
        if op == 0: # insert       
            union(i,A.popleft(), parent, rank)
        else: # extract
            num_ex += 1
            # if two extractions in a row then use a dummy set
            # to fil the gap (so that there's available for "lumping")
            if ops[j-1] == 1: 
                pos_root.append(n+num_ex-1)
                root_pos[n+num_ex-1] = len(pos_root)-1 
            # number this group of insertions
            else:
                root = find_set(i, parent)
                pos_root.append(root)
                root_pos[root] = len(pos_root)-1 
                i = A[0] # next group of extractions

    extracted = [None for i in range(num_ex)]
    for i in range(n):
        if num_ex < 1: 
            break # if no more extractions to be done
        root = find_set(i, parent) 
        if root in root_pos:
            j = root_pos[root] # find group of insertions that includes i
        else:
            continue # that number wasn't inserted
        extracted[j] = i
        num_ex -=1
        # essentially delete jth group of insertions 
        pos_root[j] = None 
        while j < len(pos_root) - 1 and pos_root[j] == None:
            j = j+1
        # and lump in with the next insertion group
        if j < len(pos_root) - 1:
            l = pos_root[j]        
            union(root,l, parent, rank)
            root_pos[find_set(l, parent)] = j                
    return extracted
```

####**Problem 6.10**

Given an array `A` of $n$ elements and a permutation $\pi$, compute $\pi\left(A\right)$ using only constant additional storage.

**Solution**

Every permutation can be decomposed into a composition of disjoint "cycles" (permutation a subset of the elements). Once the decomposition is found the cycles can be "executed" by performing a rotation of that (not necessarily contiguous) subarray. To figure out this decomposition start at index `0` and go to `A[P[0]]`, then `P[A[P[0]]]`, and so on until you get back to where you started from. To find the next cycle look for an index not included in any other cycle. 

As described this algorithm seems like it uses temporary storage to store all of the cycles, and then still leaves the question of how to perform the rotation unanswered, but you can do the rotation simultaneously with the discovery of the cycle and you can keep track of which elements have been "decomposed" with a trick: mark `P[i]` as visited.

```python
def permutation(A,P):
    for i in range(len(A)):
        if P[i] >= 0:
            a = i
            temp = A[i]
            while True: # emulates do-while
                next_a = P[a]
                next_temp = A[next_a]
                A[next_a] = temp
                # mark as visited
                P[a] -= len(P)
                a, temp = next_a, next_temp
                if a == i:
                    break
```

The problem is that this isn't really a constant space algorithm: it uses the sign bits of the entries in `P` and so is actually $O\left(n\right)$. You can do away with keeping track by only going from left to right and only applying the cycle if the entry you've reached is the leftmost position in the cycle. This takes $O\left(n\right)$ checking (you actually traverse the entire cycle to check) so the entire algorithm becomes $O\left(n^2\right)$.

```python
def permutation(A,P):
    for i in range(len(A)):
        # check if leftmost
        is_left = True
        j = P[i]
        while j != i:
            if j < i: # not leftmost
                is_min = False
                break
            j = P[j]
        
        if is_min:
            a = i
            temp = A[i]
            while True: # emulates do-while
                next_a = P[a]
                next_temp = A[next_a]
                A[next_a] = temp
                # mark as visited
                P[a] -= len(P)
                a, temp = next_a, next_temp
                if a == i:
                    break
```


####**Problem 6.11**

Same as [Problem 6.10](#problem-610) but apply $\pi^{-1}$.

**Solution**

I don't understand the solution in EPI but you can compute the inverse of any cycle $g$ by $g^{-1} = g^{\left|g\right|-1}$

####**Problem 6.12**

Given a permutation $\pi_1$ compute the next permutation in lexicographic order.

**Solution**

Suppose $\pi_1$ is represented as a vector `p`. The key insight is if for some $k$ it's the case that $p\left[k\right] < p\left[k+1\right]$ and then from there forward $p$ is non-increasing, i.e. for $i\geq k+1$ it's the case that $p\left[i\right] \geq p\left[i+1\right]$ then no rearrangement of the elements `p[i:]` could be ahead of `p` in lexicogrpahic order (because rearranging would make some more significant "digit" smaller, thereby going backwards in lexicographic order). 

So the algorithm is to find the longest non-increasing suffix, swap the least significant "digit" with the digit just beyond the suffix, then sort modified suffix. In fact you don't need to sort, just reverse, since it's already in non-increasing order. If this is confusing read the next variant.

```python
def next_permutation(P):
    # find longest non-increasing suffix
    i = len(P) - 1
    while P[i-1] >= P[i]:
        i -= 1
    # find last entry that's larger than P[i-1]
    i -= 1 # one to the left of the suffix
    j = i
    while P[j+1] >= P[i]:
        j += 1
    P[j], P[i] = P[i], P[j]
    # now reverse
    P[i+1:] = list(reversed(P[i+1:]))
```


####**Variant 6.12.0.1**

Given an array compute the next, in lexicographic order, permutation of that array.

**Solution**

The solution is exactly the same. I'm only including it here so I can include an illustration

<p align="center"><img src="next-permutation-algorithm.png" style="width: 400px;"/></p>

####**Variant 6.12.0.2**

Given a permutation compute its index in the lexicographic ordering.

**Solution**

By example: take `[3,2,1,4]`. How many permutations of `[1,2,3,4]` start with 1 or 2? It's $2\cdot 3! = 12$. Hence the index of this permutation, since it starts with 3 and not 1 nor 2, must be must larger than 12. Now how many permutations of `[2,1,4]` start with 1? It's $2!=2$. Therefore the index of this permutation must be larger than 14, since `[2,1,4]` does not start with 1. Hence `[3,2,1,4]` must be the 15th permutation of `[1,2,3,4]` in lexicographic order.

####**Variant 6.12.1**

Compute the $k$th permutation under lexicographic ordering, starting from the identity permutation

**Solution**

By example: suppose we want the 15th permutation. What can we have in the first position? If we have 1 then the maximum index will be $3! = 6$, if we have 2 then the maximum index will be $2\cdot3!=12$, and if we have 3 then the maximum index will be $3\cdot3!=18$. So the first entry must be 3. Now if the second entry is 1 what's the highest index we could have? 14. Hence the second entry must be 2, and if the third entry is 1 then the highest index is 15 (and the fourth is 4 since no choices are left). Hence `[3,2,14]` is the 15th permutation of `[1,2,3,4]`.

####**Problem 6.13**

Design a $\Theta\left(n\right)$ algorithm for rotating an array `A` of $n$ elements to the right by $i$ places. You are allowed $O\left(1)\right)$ additional storage.

**Solution**

This can be done using permutations (which is why it follows this set of problems): a rotation is a permutation and it can be broken up into cycles with the following properties:

1. All of the cycles have the same length and are shifted versions of the cycle $\left\langle 0, i \text{ mod } n, 2i \text{ mod } n, \dots, (\left(l-1\right)i \text{ mod } n \right\rangle$.

2. The number of cycles is $\gcd\left(n,i\right)$.
 
So you can use the solution to [Problem 6.10](#problem-610) (good luck) or you can just realize the rotation flips two parts of the array. Suppose the array is `A = [1,2,3,4,a,b]` and $i=2$. Well obviously the result is reversing the order of `A[0:4]` and `A[4:]`. The only trick is how to do this in place: use reverse.

```python
def rotate_arry(A,i):
    i %= len(A)
    A = A[::-1] # [1,2,3,4,a,b] -> [b,a,4,3,2,1]
    A[:i] = list(reverse(A[:i])) # [b,a] -> [a,b]
    A[i:] = list(reverse(A[i:])) # [4,3,2,1] -> [1,2,3,4]
```

####**Problem 6.14**

Check for the validity of a Sudoku solution (0 indicates empty).

**Solution**

The only tricky thing here is checking all of the $3\times 3$ smaller squares and checking to make sure no number is repeated (or omitted). 

```python
def check_sudoku(A):
    #check the rows
    #check the columns
    for i in range(0,9,3):
        for j in range(0,9,3):
        nums = [False for i in range(10)]
            for k in range(3):
                for l in range(3):
                    entry = A[i+k][j+l]
                    if entry == 0 or nums[entry] == True:
                        return False
                    else:
                        nums[entry] = True
    return True
```                    
    
####**Problem 6.15**

Implement a function that takes a 2D array `A` and prints `A` in spiral order.

**Solution**

There's a very clever solution only possible in python.

```python
import itertools
def rotate90_ccw_and_yield(A):
    while A:
        yield A[0]
        # rotate 90 degrees ccw = transpose and flipud
        A = list(reversed(list(zip(*A[1:]))))

# turn iterator/generator into a list
print(list(itertools.chain(*rotate90_ccw_and_yield(A))))
```   

####**Variant 6.15.1**

Given a dimension $d$, write a program to generate a $d \times d$ array which when printed in spiral order output gives $\left\langle 1,2,\dots,d^2\right\rangle$.

**Solution**

There's a very clever solution only possible in python.

```python
import itertools
def rotate90_ccw_and_yield(A):
    while A:
        yield A[0]
        # rotate 90 degrees ccw = transpose and flipud
        A = list(reversed(list(zip(*A[1:]))))

# turn iterator/generator into a list
print(list(itertools.chain(*rotate90_ccw_and_yield(A))))
```   




Given and 
-
-
-
-
-
-
-
-
-
-
adfsdf
