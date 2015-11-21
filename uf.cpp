/*
 * Adapted from kartik kukreja's implementation available at:
 * https://github.com/kartikkukreja/blog-codes/blob/master/src/Union%20Find%20(Disjoint%20Set)%20Data%20Structure.cpp
 */

#include "uf.hpp"
#include <vector>

using namespace std;

// Create an empty union find data structure with N isolated sets.
UF::UF(int N)   {
    cacheSize = N;
    id = vector<int>(N);
    sz = vector<int>(N);
    for(int i=0; i<N; i++)	{
        id[i] = i;
        sz[i] = 1;
    }
}

// Return the id of component corresponding to object p.
int UF::find(int p)	{
    // If p is not accommodated by the cache
    if(p >= cacheSize) {
        // Resize this structure so it can accommodate p
        resize(2*p);
    }
    int root = p;
    while (root != id[root])
        root = id[root];
    while (p != root) {
        int newp = id[p];
        id[p] = root;
        p = newp;
    }
    return root;
}

// Replace sets containing x and y with their union.
void UF::merge(int x, int y)	{
    int i = find(x);
    int j = find(y);
    if (i == j) return;

    // make smaller root point to larger one
    if   (sz[i] < sz[j])	{
        id[i] = j;
        sz[j] += sz[i];
    } else	{
        id[j] = i;
        sz[i] += sz[j];
    }
}

// Are objects x and y in the same set?
bool UF::connected(int x, int y)    {
    return find(x) == find(y);
}

// Increase the size of possible classes
void UF::resize(int newSize) {
    int oldSize = cacheSize;
    id.resize(newSize);
    sz.resize(newSize);
    for(int i = oldSize; i < newSize; i++) {
        id[i] = i;
        sz[i] = 1;
    }
    cacheSize = newSize;
}
