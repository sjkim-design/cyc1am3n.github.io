---
layout: post
title:  "[Daily PS] 파이썬으로 구현하는 BFS와 DFS"
subtitle: "BFS and DFS with Python"
post_description: "파이썬으로 BFS와 DFS를 구현하는 내용입니다."
date:   2019-04-26 15:40:54 +0900
tags: [algorithm]
background: 'https://images.unsplash.com/photo-1484788984921-03950022c9ef?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=2904&q=80'
author: cyc1am3n
comments: true
---

### 시작하기 앞서..

최근 코딩 테스트를 볼 일이 생겨 일주일 간 벼락치기로 PS(Problem Solving) 공부를 했었는데요, 수업 때 배운 내용 중 기억나는 건 "동적 계획법은 메모이제이션(memoization)이다." 정도 밖에 없어서 정말 걱정하면서 시작했었네요.

그래도 코딩 테스트에 자주 출제되는 유형 위주로 열심히 준비해가니 시험 볼 때 그렇게 멘붕하지는 않았던 것 같습니다.

아무튼 이번 계기로 PS 공부를 꾸준히 해야 되겠다 라는 생각을 하게 되어 거창한 이름이지만 매일 공부하자는 다짐과 함께 Daily PS 라는 포스팅을 시작하게 되었습니다. (매일 올린다는 이야기는 아닙니다..ㅎ)

요즘은 파이썬으로만 코딩을 해서 벼락치기하는 동안 다른 언어를 다시 복습 할 겨를이 없어 파이썬으로 준비를 했었는데, 일단 당분간은 파이썬으로 PS를 할 생각입니다. 어차피 구현 아이디어는 같으니깐 다른 것도 금방하지 않을까요..?

---

<br />

## BFS(Breadth First Search, 너비 우선 탐색)

일단 오늘 다뤄볼 내용인 BFS와 DFS는 모두 그래프를 탐색할 때 사용하는 기법입니다. 이름에서도 알 수 있듯이 어떤 것을 우선 순위로 하는지 차이라서 코드도 거의 비슷하게 느껴지실 겁니다.

일단 너비 우선 탐색이라고 불리는 BFS는 말 그대로 너비를 우선해서 그래프를 탐색하는 기법인데요, 시작점인 루트 노드와 같은 거리에 있는 노드를 우선으로 방문한다고 보시면 됩니다.

아래 그림을 보시면 확실하게 이해하실 수 있을 것 같네요.

{% include image.html file="https://upload.wikimedia.org/wikipedia/commons/5/5d/Breadth-First-Search-Algorithm.gif" description="BFS, Wikimedia Commons" class="center-50"%}

이 알고리즘의 핵심은 **큐(queue)** 자료구조를 사용하는 것인데요, 노드를 방문하면서 인접한 노드 중 방문하지 않았던 노드의 정보만 큐에 넣어 먼저 큐에 들어있던 노드부터 방문하면 되는 것이죠. 물론 큐를 사용하지 않아도 구현이 가능합니다!

한편, 파이썬에서 큐를 `list` 타입을 사용해 자료를 입력할 때는 `list.append(something)`, 출력할 때는 `list.pop(0)` 와 같이 구현하시는 분들이 있습니다.

하지만 `list.pop(0)` 은 시간복잡도가 **O(N)** 이라 이렇게 구현하면 시간적으로 매우 비효율적인 코드가 만들어지게 됩니다. [[링크](https://wiki.python.org/moin/TimeComplexity)]

따라서 `collections` 라이브러리의 `deque` 를 사용하면 시간을 절약할 수 있게 됩니다.

또한 인접 노드 중 방문하지 않았던 노드를 큐에 넣을 때는 파이썬 데이터 타입 중 `set` 을 사용하면 아주 쉽게 구현할 수 있습니다.

만약 다음과 같이 방향이 있는 유향그래프를 BFS로 탐색한다면,

```python
graph_list = {1: set([3, 4]),
              2: set([3, 4, 5]),
              3: set([1, 5]),
              4: set([1]),
              5: set([2, 6]),
              6: set([3, 5])}
root_node = 1
```

이렇게 구현하시면 됩니다. 

```python
from collections import deque

def BFS_with_adj_list(graph, root):
    visited = []
    queue = deque([root])

    while queue:
        n = queue.popleft()
        if n not in visited:
            visited.append(n)
            queue += graph[n] - set(visited)
    return visited
  
print(BFS_with_adj_list(graph_list, root_node))
```



---

<br />

## DFS(Depth First Search, 깊이 우선 탐색)

DFS는 BFS와는 다르게 한 놈만 팬다(?)라는 느낌으로 한 방향으로 갈 수 있을 만큼 깊게 탐색한다는 의미에서 깊이 우선 탐색이라는 이름이 붙었습니다.

갈 수 있는 한 끝까지 탐색해 리프 노드를 방문하고, 이전 갈림길에서 선택하지 않았던 노드를 방문하는 식으로 탐색합니다.

이번에도 설명 대신에 이미지로 보시는게 더 확실하게 이해되실 것 같네요.

{% include image.html file="https://upload.wikimedia.org/wikipedia/commons/7/7f/Depth-First-Search.gif" description="DFS, Wikimedia Commons" class="center-50"%}

한편, 여기에서는 BFS에 있던 큐 대신에 **스택(stack)** 으로 자료구조를 대체하기만 하면 쉽게 구현하실 수 있습니다.

먼저 방문한 노드에 연결된 노드보다 현재 방문한 노드에 연결된 노드를 방문해야 한 방향으로 갈 수 있거든요.

이전과 같은 유향 그래프를 탐색하신다면 이렇게 구현하시면 됩니다.

```python
def DFS_with_adj_list(graph, root):
    visited = []
    stack = [root]

    while stack:
        n = stack.pop()
        if n not in visited:
            visited.append(n)
            stack += graph[n] - set(visited)
    return visited

print(BFS_with_adj_list(graph_list, root_node))
```

그럼 오늘의 포스팅은 예제 문제 2개를 풀어보며 마무리 하겠습니다!

---

<br />

### 예제 문제 1

이를 활용한 문제가 가장 간단한 문제가 백준 1260번 [DFS와 BFS (링크)](https://www.acmicpc.net/problem/1260) 문제가 아닐까 싶은데요, 자세한 내용은 다음과 같습니다.

{% include image.html file="https://user-images.githubusercontent.com/11629647/56722544-43a7f180-6782-11e9-94a8-0186cb06dfcf.png" class="center-95"%}

이 문제에서는 위에서 구현했던 알고리즘에 정렬 기능만 추가해주면 됩니다. (스택/큐에 넣을 때 정렬 방향이 다르다는 것을 주의!)

또 여기의 그래프는 유향그래프가 아닌 양방향 그래프이므로 이 점에 유의해서 그래프를 만드시면 되겠습니다.

```python
from collections import deque

def DFS(graph, root):
    visited = []
    stack = [root]

    while stack:
        n = stack.pop()
        if n not in visited:
            visited.append(n)
            if n in graph:
                temp = list(set(graph[n]) - set(visited))
                temp.sort(reverse=True)
                stack += temp
    return " ".join(str(i) for i in visited)

def BFS(graph, root):
    visited = []
    queue = deque([root])

    while queue:
        n = queue.popleft()
        if n not in visited:
            visited.append(n)
            if n in graph:
                temp = list(set(graph[n]) - set(visited))
                temp.sort()
                queue += temp
    return " ".join(str(i) for i in visited)

  
graph = {}
n = input().split(' ')
node, edge, start = [int(i) for i in n]
for i in range(edge):
    edge_info = input().split(' ')
    n1, n2 = [int(j) for j in edge_info]
    if n1 not in graph:
        graph[n1] = [n2]
    elif n2 not in graph[n1]:
        graph[n1].append(n2)

    if n2 not in graph:
        graph[n2] = [n1]
    elif n1 not in graph[n2]:
        graph[n2].append(n1)

print(DFS(graph, start))
print(BFS(graph, start))
```

---

<br />

### 예제 문제 2

이 문제는 이번에 봤던 코딩 테스트에서 나온 문제입니다.

다음과 같이 `Tree` class가 정의되어 있을 때, 

```python
class Tree(object):
    def __init__(self, x, l=None, r=None): # 'None' means empty Node
        self.x = x	# value of Node
        self.l = l	# left child of Node
        self.r = r	# right child of Node
```

이렇게 생긴 트리는

```python
T = Tree(4, Tree(5, Tree(4, Tree(5, None, None), None), None), Tree(6, Tree(1, None, None), Tree(6, None, None)))
```

다음과 같은 모양을 가집니다.

{% include image.html file="https://user-images.githubusercontent.com/11629647/56787599-fede9200-6837-11e9-945c-6c56e0f32f7a.png" class="center-50"%}

아무튼 여기에서 해결해야하는 것은 루트 노드부터 리프 노드까지의 경로 중 **가장 다양한 값을 가진 경로에서 볼 수 있는 값의 갯수**를 구하는 것 입니다.

예를 들어, 위의 트리에서 볼 수 있는 경로는 [4, 5, 4, 5], [4, 6, 1], [4, 6, 6] 이렇게 세 가지 인데 각 경로당 볼 수 있는 값의 갯수는 2(4, 5), 3(1, 4, 6), 2(4, 6)이므로 이 트리를 입력으로 받게 되면 3을 출력하면 되겠습니다.

이 문제에서는 **DFS**를 이용하면 쉽게 풀 수 있는데요, 스택에 **[방문하려는 노드, 지금까지의 경로, 경로 중 볼 수 있는 서로 다른 값]**을 넣어주면 리프 노드를 방문했을 때 이 스택의 세 번째 원소를 사용하면 최댓값이 어떻게 되는지 쉽게 구할 수 있습니다. 여기에서도 중복되는 값은 알아서 걸러주는 `set` 덕분에 중복 신경쓰지 않고 방문할 때마다 값을 집어 넣을 수 있었습니다.

여기에 대한 해답은 다음과 같습니다.

```python
def solution(T):
    distinct = {1: set([])}
    stack = [(T, [T], set([T.x]))]
    i = 1  # number of path
    while stack:  # DFS
        n, path, value = stack.pop()
        if n.l == None and n.r == None:  # leaf node
            distinct[i] = value
            i = i + 1
        else:
            if n.r != None:
                stack.append((n.r, path + [n.r], value | set([n.r.x])))
            if n.l != None:
                stack.append((n.l, path + [n.l], value | set([n.l.x])))

    answer = 1

    for key in distinct.keys():
        temp = len(distinct[key])
        if temp > answer:
            answer = temp
    print(distinct)
    return answer
```



---

<br />