#ifndef PDB_QUERYINTERMEDIARYREP_SETEXPRESSIONIR_H
#define PDB_QUERYINTERMEDIARYREP_SETEXPRESSIONIR_H

#include "MaterializationMode.h"
#include "MaterializationModeNone.h"
#include "SetExpressionIrAlgo.h"

using std::make_shared;
using std::shared_ptr;
using std::string;

namespace pdb_detail {
/**
 * Base class for any class that models a PDB set.
 */
class SetExpressionIr {

public:
    /**
     * Executes the given algorithm on the expression.
     *
     * @param algo the algoithm to execute.
     */
    virtual void execute(SetExpressionIrAlgo& algo) = 0;

    virtual string getName() = 0;

    void setMaterializationMode(shared_ptr<MaterializationMode> materializationMode) {
        _materializationMode = materializationMode;
    }

    shared_ptr<MaterializationMode> getMaterializationMode() {
        return _materializationMode;
    }

    // added for converting logical plan into physical plan

    void setTraversed(bool traversed, int traversalId) {
        _traversed = traversed;
        _traversalId = traversalId;
    }

    bool isTraversed() {
        return _traversed;
    }

    int getTraversalId() {
        return _traversalId;
    }


private:
    shared_ptr<MaterializationMode> _materializationMode = make_shared<MaterializationModeNone>();

    // added for converting logical plan into physical plan

    bool _traversed = false;

    int _traversalId = -1;
};

typedef shared_ptr<SetExpressionIr> SetExpressionIrPtr;
}

#endif  // PDB_QUERYINTERMEDIARYREP_SETEXPRESSIONIR_H
