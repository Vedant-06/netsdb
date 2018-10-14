#ifndef PDB_TCAPPARSER_FILTEROPERATION_H
#define PDB_TCAPPARSER_FILTEROPERATION_H

#include <memory>
#include <string>

#include "BuildTcapIrTests.h"
#include "RetainClause.h"
#include "TableExpression.h"
#include "TcapIdentifier.h"
#include "TcapTokenStream.h"

using std::shared_ptr;
using std::string;

namespace pdb_detail {
/**
 * Models a filter opertion in the TCAP grammar.  For example:
 *
 *    filter D by isExamGreater retain all
 *
 * In this example:
 *
 *     inputTableName would be D
 *     filterColumnName would be isExamGreater
 *     an instance of RetailAllClause would be the value of retain
 */
class FilterOperation : public TableExpression {
public:
    /**
     * The name of the table filterd by the operation.
     */
    const TcapIdentifier inputTableName;

    /**
     * The name of the column in inputTableName to be filtered upon.
     */
    const TcapIdentifier filterColumnName;

    /**
     * The retention clause of the operation.
     */
    const shared_ptr<RetainClause> retain;

    // contract from super
    void match(function<void(LoadOperation&)>,
               function<void(ApplyOperation&)>,
               function<void(FilterOperation&)> forFilter,
               function<void(HoistOperation&)>,
               function<void(BinaryOperation&)>) override;

private:
    /**
     * Creates a new FilterOperation.
     *
     * If retain is nullptr, throws invalid_argument exception.
     *
     * @param inputTableName The name of the table filterd by the operation.
     * @param filterColumnName The name of the column in inputTableName to be filtered upon.
     * @param retain The retention clause of the operation.
     * @return a new FilterOperation
     */
    // private because throws exception and PDB styel guide forbids exceptions crossing API
    // boundaries
    FilterOperation(TcapIdentifier inputTableName,
                    TcapIdentifier filterColumnName,
                    shared_ptr<RetainClause> retain);

    friend shared_ptr<FilterOperation> makeFilterOperation(
        TcapTokenStream& tokens);  // for constructor

    friend void pdb_tests::buildTcapIrTest4(class UnitTest& qunit);  // for constructor
};

typedef shared_ptr<FilterOperation> FilterOperationPtr;
}

#endif  // PDB_TCAPPARSER_FILTEROPERATION_H
