
#ifndef JOIN_MAP_H
#define JOIN_MAP_H

// PRELOAD %JoinMap <Nothing>%

#include "Object.h"
#include "Handle.h"
#include "JoinPairArray.h"

template <typename StoredType>
class JoinRecordLst;

namespace pdb {

// This is the Map type used to power joins

template <class ValueType>
class JoinMap : public Object {

private:
    // this is where the data are actually stored
    Handle<JoinPairArray<ValueType>> myArray;

    // size of ValueType
    size_t objectSize;

    // my partition id
    size_t partitionId;

    // number of partitions (per node)
    int numPartitions;

public:
    ENABLE_DEEP_COPY

    // this constructor creates a map with specified slots, partitionId and numPartitions
    JoinMap(uint32_t initSize, size_t partitionId, int numPartitions);

    // this constructor pre-allocates initSize slots... initSize must be a power of two
    JoinMap(uint32_t initSize);

    // this constructor creates a map with a single slot
    JoinMap();

    // destructor
    ~JoinMap();

    // allows us to access all of the records with a particular hash value
    JoinRecordList<ValueType> lookup(const size_t& which);

    // adds a new value at position which
    ValueType& push(const size_t& which);

    // clears the particular key from the map, destructing both the key and the value.
    // This is typically used when an out-of-memory
    // exception is thrown when we try to add to the hash table, and we want to immediately clear
    // the last item added.
    void setUnused(const size_t& clearMe);

    // returns the number of elements in the map
    size_t size() const;

    // returns 0 if this entry is undefined; 1 if it is defined
    int count(const size_t& which);

    // these are used for iteration
    JoinMapIterator<ValueType> begin();
    JoinMapIterator<ValueType> end();


    // JiaNote: add partition id to enable hash partitioned join
    size_t getPartitionId();
    void setPartitionId(size_t partitionId);
    int getNumPartitions();
    void setNumPartitions(int numPartitions);

    // JiaNote: add this to enable combination of two JoinMaps
    size_t getObjectSize();
    void setObjectSize();
};
}

#include "JoinMap.cc"

#endif
