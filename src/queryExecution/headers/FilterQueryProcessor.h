
#ifndef FILTER_QUERY_PROCESSOR_H
#define FILTER_QUERY_PROCESSOR_H

#include "SimpleSingleTableQueryProcessor.h"
#include "UseTemporaryAllocationBlock.h"
#include "InterfaceFunctions.h"
#include "PDBVector.h"
#include "Selection.h"
#include "Handle.h"

namespace pdb {

template <class Output, class Input>
class FilterQueryProcessor : public SimpleSingleTableQueryProcessor {

private:
    // this is where we write the results
    UseTemporaryAllocationBlockPtr blockPtr;

    // this is where the input objects are put
    Handle<Input> inputObject;

    // this is the list of input objects
    Handle<Vector<Handle<Input>>> inputVec;

    // this is where we are in the input
    size_t posInInput;

    // this is where the output objects are put
    Handle<Vector<Handle<Input>>> outputVec;

    // and here are the lamda objects used to proces the input vector
    SimpleLambda<bool> filterPred;

    // and here are the actual functions
    std::function<bool()> filterFunc;

    // tells whether we have been finalized
    bool finalized;

public:
    FilterQueryProcessor(Selection<Output, Input>& forMe);
    FilterQueryProcessor(SimpleLambda<bool> filterPred);
    // the standard interface functions
    void initialize() override;
    void loadInputPage(void* pageToProcess) override;
    void loadOutputPage(void* pageToWriteTo, size_t numBytesInPage) override;
    bool fillNextOutputPage() override;
    void finalize() override;
    void clearOutputPage() override;
    void clearInputPage() override;
};
}

#include "FilterQueryProcessor.cc"

#endif
