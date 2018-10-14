#ifndef PDB_QUERYINTERMEDIARYREP_RECORDPREDICATEIR_H
#define PDB_QUERYINTERMEDIARYREP_RECORDPREDICATEIR_H

#include "Handle.h"
#include "Lambda.h"
#include "Object.h"
#include "QueryNodeIr.h"

using pdb::Handle;
using pdb::Lambda;
using pdb::Object;

namespace pdb_detail {
/**
 * A boolean function that operates over a single input record.
 */
class RecordPredicateIr {

public:
    /**
     * Produces a Lambda<bool> representation of the predicate.
     *
     * @param inputRecordPlaceholder a placeholder to represend the "free variable" input record the
     * predicate
     *                               operates over within the structure of the returned Lambda.
     * @return a Lambda version of the predciate.
     */
    virtual Lambda<bool> toLambda(Handle<Object>& inputRecordPlaceholder) = 0;
};
}

#endif  // PDB_QUERYINTERMEDIARYREP_RECORDPREDICATEIR_H
