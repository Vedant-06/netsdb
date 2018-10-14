
#ifndef FILTER_BLOCK_QUERY_PROCESSOR_H
#define FILTER_BLOCK_QUERY_PROCESSOR_H

// by Jia, Oct 2016

#include "BlockQueryProcessor.h"
#include "UseTemporaryAllocationBlock.h"
#include "InterfaceFunctions.h"
#include "PDBVector.h"
#include "Selection.h"
#include "Handle.h"
#include "GenericBlock.h"

namespace pdb {

template <class Output, class Input>
class FilterBlockQueryProcessor : public BlockQueryProcessor {

private:
    // this is where the input objects are put
    Handle<Output> inputObject;

    // this is the list of input objects
    Handle<GenericBlock> inputBlock;

    // this is where we are in the input
    size_t posInInput;

    // this is where the output objects are put
    Handle<GenericBlock> outputBlock;

    // and here are the lamda objects used to proces the input vector
    SimpleLambda<bool> filterPred;

    // and here are the actual functions
    std::function<bool()> filterFunc;

    // tells whether we have been finalized
    bool finalized;

    // output batch size
    size_t batchSize;

public:
    ~FilterBlockQueryProcessor();
    FilterBlockQueryProcessor(Selection<Output, Input>& forMe);
    FilterBlockQueryProcessor(SimpleLambda<bool> filterPred);
    // the standard interface functions
    void initialize() override;
    void loadInputBlock(Handle<GenericBlock> block) override;
    Handle<GenericBlock>& loadOutputBlock() override;
    bool fillNextOutputBlock() override;
    void finalize() override;
    void clearOutputBlock() override;
    void clearInputBlock() override;
};
}

#include "FilterBlockQueryProcessor.cc"

#endif
