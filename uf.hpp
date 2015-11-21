#ifndef UF_H
#define UF_H

#include <vector>

class UF    {
    std::vector<int> id;
    std::vector<int> sz;
    int cacheSize;
public:
    // Create an empty union find data structure that can store up to N isolated sets.
    UF(int N);

    // Return the id of component corresponding to object p.
    int find(int p);
    // Replace sets containing x and y with their union.
    void merge(int x, int y);
    // Are objects x and y in the same set?
    bool connected(int x, int y);
    // Increase the cache size for this structure
    void resize(int newSize);
};

#endif // UF_H
