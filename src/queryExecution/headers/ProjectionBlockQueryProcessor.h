
#ifndef PROJECTION_BLOCK_QUERY_PROCESSOR_H
#define PROJECTION_BLOCK_QUERY_PROCESSOR_H

#include "SimpleSingleTableQueryProcessor.h"
#include "UseTemporaryAllocationBlock.h"
#include "InterfaceFunctions.h"
#include "PDBVector.h"
#include "Selection.h"
#include "Handle.h"
#include "GenericBlock.h"

namespace pdb {

template <class Output, class Input>
class ProjectionBlockQueryProcessor : public BlockQueryProcessor {

private:
    // this is where the input objects are put
    Handle<Input> inputObject;

    // this is the list of input objects
    Handle<GenericBlock> inputBlock;

    // this is where we are in the input
    size_t posInInput;

    // this is where the output objects are put
    Handle<GenericBlock> outputBlock;

    // and here are the lamda objects used to proces the input vector
    SimpleLambda<Handle<Output>> projection;

    // and here are the actual functions
    std::function<Handle<Output>()> projectionFunc;

    // tells whether we have been finalized
    bool finalized;

    // output batch size
    size_t batchSize;

public:
    ~ProjectionBlockQueryProcessor();
    ProjectionBlockQueryProcessor(Selection<Output, Input>& forMe);
    ProjectionBlockQueryProcessor(SimpleLambda<Handle<Output>> projection);
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

#include "ProjectionBlockQueryProcessor.cc"

#endif
