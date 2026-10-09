"""Bounded 2D A* with inflated obstacle footprints; no-path is a valid result."""
import heapq
import numpy as np


def plan_route(start,goal,obstacles,resolution=.08,margin=.18,bound=2.2):
    if np.any(np.abs(start)>bound) or np.any(np.abs(goal)>bound):
        raise ValueError('Destination is outside the navigation workspace')
    axis=np.arange(-bound,bound+resolution/2,resolution)
    X,Y=np.meshgrid(axis,axis,indexing='ij')
    blocked=np.zeros(X.shape,dtype=bool)
    for o in obstacles:
        if abs(o[6])>1e-8:
            raise ValueError('Navigation planner requires axis-aligned boxes')
        blocked|=(np.abs(X-o[0])<=o[3]+margin)&(np.abs(Y-o[1])<=o[4]+margin)
    cell=lambda p:tuple(np.rint((np.asarray(p)+bound)/resolution).astype(int))
    a,b=cell(start),cell(goal)
    valid=lambda p:0<=p[0]<len(axis) and 0<=p[1]<len(axis) and not blocked[p]
    if not valid(a) or not valid(b):
        raise ValueError('Start or goal is blocked/outside workspace')
    costs={a:0.};parent={};queue=[(0.,a)]
    while queue:
        _,p=heapq.heappop(queue)
        if p==b:
            break
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
            q=(p[0]+dx,p[1]+dy)
            if not valid(q) or dx and dy and (not valid((p[0]+dx,p[1])) or not valid((p[0],p[1]+dy))):
                continue
            score=costs[p]+np.hypot(dx,dy)
            if score<costs.get(q,float('inf')):
                costs[q]=score;parent[q]=p
                heapq.heappush(queue,(score+np.hypot(q[0]-b[0],q[1]-b[1]),q))
    if b not in costs:
        raise ValueError('No collision-free route in bounded workspace')
    path=[b]
    while path[-1]!=a:
        path.append(parent[path[-1]])
    points=[np.asarray(start),*[np.array([axis[i],axis[j]]) for i,j in reversed(path)],np.asarray(goal)]
    def clear(u,v):
        samples=np.linspace(u,v,max(2,int(np.linalg.norm(v-u)/(.5*resolution))+2))
        return all(not np.any((np.abs(samples[:,0]-o[0])<=o[3]+margin)&(np.abs(samples[:,1]-o[1])<=o[4]+margin)) for o in obstacles)
    simplified=[points[0]];i=0
    while i<len(points)-1:
        j=len(points)-1
        while j>i+1 and not clear(points[i],points[j]):
            j-=1
        simplified.append(points[j]);i=j
    if not all(clear(a,b) for a,b in zip(simplified,simplified[1:])):
        raise ValueError('Grid route cannot satisfy continuous clearance')
    return np.asarray(simplified)
