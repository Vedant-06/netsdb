#ifndef PDB_TCAPPARSER_APPLYOPERATION_H
#define PDB_TCAPPARSER_APPLYOPERATION_H

#include <memory>
#include <string>
#include <vector>

#include "BuildTcapIrTests.h"
#include "RetainClause.h"
#include "TableExpression.h"
#include "TcapIdentifier.h"
#include "TcapTokenStream.h"

using std::shared_ptr;
using std::string;
using std::vector;

namespace pdb_detail {
/**
 * Whether the apply operation was of the form "apply func" or "apply method"
 */
enum ApplyOperationType { func, method };

/**
 * Models an apply opertion in the TCAP grammar.  For example:
 *
 *     apply func "avgExams" to A[student] retain all
 *
 * In this example:
 *
 *     avgExams would be the applyTarget
 *     func would be the applyType
 *     A would be the input table
 *     [student] would be the inputTableColumnNames
 *     an instance of RetailAllClause would be the value of retain
 */
class ApplyOperation : public TableExpression {
public:
    /**
     * The name of the function or method to be applied. This is just metadata and the value has no
     * relationship
     * to TCAP itself.
     */
    const string applyTarget;

    /**
     * Indicates whether the func or method keyword was used in the apply statement.
     *
     * This is just metadata and the value has no relationship to TCAP itself.
     */
    const ApplyOperationType applyType;

    /**
     * The table from which input to the apply statement is drawn.
     */
    const TcapIdentifier inputTable;

    /**
     * The specific columns in the input table from which the input to the apply operation is drawn.
     */
    const shared_ptr<const vector<TcapIdentifier>> inputTableColumnNames;

    /**
     * The retention clause of the operation.
     */
    const shared_ptr<RetainClause> retain;


    // contract from super
    void match(function<void(LoadOperation&)>,
               function<void(ApplyOperation&)> forApply,
               function<void(FilterOperation&)>,
               function<void(HoistOperation&)>,
               function<void(BinaryOperation&)>) override;

private:
    /**
     * Creates a new ApplyOperation.
     *
     * Throws invalid_argument if inputTableColumnNames or retain is nullptr
     *
     * @param applyTarget The name of the function or method to be applied.
     * @param applyType Indicates whether the func or method keyword was used in the apply
     * statement.
     * @param inputTable The table from which input to the apply statement is drawn.
     * @param inputTableColumnNames The specific columns in the input table from which the input to
     * the apply operation is drawn.
     * @param retain The retention clause of the operation.
     * @return the ApplyOperation instance.
     */
    // private because throws exception and PDB style guide disallows exceptions across API
    // boundaries
    ApplyOperation(string applyTarget,
                   ApplyOperationType applyType,
                   TcapIdentifier inputTable,
                   shared_ptr<vector<TcapIdentifier>> inputTableColumnNames,
                   shared_ptr<RetainClause> retain);

    friend shared_ptr<ApplyOperation> makeApplyOperation(
        TcapTokenStream& tokens);  // for constructor

    friend void pdb_tests::buildTcapIrTest2(class UnitTest& qunit);  // for constructor

    friend void pdb_tests::buildTcapIrTest3(class UnitTest& qunit);  // for constructor
};

typedef shared_ptr<ApplyOperation> ApplyOperationPtr;
}

#endif  // PDB_TCAPPARSER_APPLYOPERATION_H
