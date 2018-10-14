
#ifndef BLOCK_QUERY_PROCESSOR_H
#define BLOCK_QUERY_PROCESSOR_H

#include <memory>
#include "GenericBlock.h"
#include "PipelineContext.h"

namespace pdb {

class BlockQueryProcessor;
typedef std::shared_ptr<BlockQueryProcessor> BlockQueryProcessorPtr;

// this pure virtual class is spit out by a simple query class (like the Selection class)... it is
// then
// used by the system to process queries
//
class BlockQueryProcessor {

public:
    // must be called before the query processor is asked to do anything
    virtual void initialize() = 0;

    // loads up another input block to read input data
    virtual void loadInputBlock(Handle<GenericBlock> block) = 0;

    // load up another output block to write output data
    virtual Handle<GenericBlock>& loadOutputBlock() = 0;

    // attempts to fill the next output block with data.  Returns true if it can.  If it
    // cannot, returns false, and the next call to loadInputBlock should be made
    virtual bool fillNextOutputBlock() = 0;

    // must be called after all of the input pages have been sent in
    virtual void finalize() = 0;

    // must be called before free the data in output page
    virtual void clearOutputBlock() = 0;

    // must be called before free the data in input page
    virtual void clearInputBlock() = 0;

    // set pipeline context
    virtual void setContext(PipelineContextPtr context) {
        this->context = context;
    }

    // get pipeline context
    PipelineContextPtr getContext() {
        return this->context;
    }

protected:
    PipelineContextPtr context;
};
}

#endif
