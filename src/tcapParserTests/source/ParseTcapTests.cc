#ifndef PDB_TCAPPARSERTESTS_PARSETCAPTESTS_H
#define PDB_TCAPPARSERTESTS_PARSETCAPTESTS_H


#include "ParseTcapTests.h"

#include <string>

#include "ApplyOperation.h"
#include "FilterOperation.h"
#include "LoadOperation.h"
#include "GreaterThanOp.h"
#include "HoistOperation.h"
#include "RetainAllClause.h"
#include "RetainExplicitClause.h"
#include "RetainNoneClause.h"
#include "StoreOperation.h"
#include "TableAssignment.h"
#include "TcapParser.h"
#include "qunit.h"

using std::string;

using pdb_detail::TranslationUnit;
using pdb_detail::ApplyOperation;
using pdb_detail::ApplyOperationType;
using pdb_detail::FilterOperation;
using pdb_detail::LoadOperation;
using pdb_detail::StoreOperation;
using pdb_detail::HoistOperation;
using pdb_detail::BinaryOperation;
using pdb_detail::GreaterThanOp;
using pdb_detail::TableAssignment;
using pdb_detail::RetainAllClause;
using pdb_detail::RetainNoneClause;
using pdb_detail::RetainExplicitClause;
using pdb_detail::parseTcap;

using QUnit::UnitTest;

namespace pdb_tests {
void testParseTcap1Help(UnitTest& qunit, TranslationUnit parseTree) {
    int statementNumber = 0;

    parseTree.statements->operator[](statementNumber++)
        ->match(
            [&](TableAssignment& assignment) {
                QUNIT_IS_EQUAL(1, assignment.attributes->size());

                QUNIT_IS_EQUAL("exec", assignment.attributes->operator[](0).name.contents);
                QUNIT_IS_EQUAL("\"exec1\"", assignment.attributes->operator[](0).value.contents);

                QUNIT_IS_EQUAL("A", assignment.tableName.contents)

                QUNIT_IS_EQUAL(1, assignment.columnNames->size());

                QUNIT_IS_EQUAL("student", assignment.columnNames->operator[](0).contents)

                assignment.value->match(
                    [&](LoadOperation& load) {
                        QUNIT_IS_EQUAL("\"(databaseName, inputSetName)\"", load.source)

                    },
                    [&](ApplyOperation&) { QUNIT_IS_TRUE(false); },
                    [&](FilterOperation&) { QUNIT_IS_TRUE(false); },
                    [&](HoistOperation&) { QUNIT_IS_TRUE(false); },
                    [&](BinaryOperation&) { QUNIT_IS_TRUE(false); });
            },
            [&](StoreOperation&) { QUNIT_IS_TRUE(false); });

    parseTree.statements->operator[](statementNumber++)
        ->match(
            [&](TableAssignment& assignment) {
                QUNIT_IS_EQUAL("B", assignment.tableName.contents)

                QUNIT_IS_EQUAL(2, assignment.columnNames->size());

                QUNIT_IS_EQUAL("student", assignment.columnNames->operator[](0).contents);
                QUNIT_IS_EQUAL("examAverage", assignment.columnNames->operator[](1).contents)

                assignment.value->match(
                    [&](LoadOperation&) { QUNIT_IS_TRUE(false); },
                    [&](ApplyOperation& applyOperation) {
                        QUNIT_IS_EQUAL(ApplyOperationType::func, applyOperation.applyType);

                        QUNIT_IS_EQUAL("\"avgExams\"", applyOperation.applyTarget)

                        QUNIT_IS_EQUAL("A", applyOperation.inputTable.contents)

                        QUNIT_IS_EQUAL(1, applyOperation.inputTableColumnNames->size());

                        QUNIT_IS_EQUAL(
                            "student",
                            applyOperation.inputTableColumnNames->operator[](0).contents);

                        QUNIT_IS_TRUE(applyOperation.retain->isAll());
                    },
                    [&](FilterOperation&) { QUNIT_IS_TRUE(false); },
                    [&](HoistOperation&) { QUNIT_IS_TRUE(false); },
                    [&](BinaryOperation&) { QUNIT_IS_TRUE(false); });
            },
            [&](StoreOperation&) { QUNIT_IS_TRUE(false); });

    parseTree.statements->operator[](statementNumber++)
        ->match(
            [&](TableAssignment& assignment) {
                QUNIT_IS_EQUAL("C", assignment.tableName.contents)

                QUNIT_IS_EQUAL(3, assignment.columnNames->size());

                QUNIT_IS_EQUAL("student", assignment.columnNames->operator[](0).contents);
                QUNIT_IS_EQUAL("examAverage", assignment.columnNames->operator[](1).contents)
                QUNIT_IS_EQUAL("hwAverage", assignment.columnNames->operator[](2).contents)

                assignment.value->match(
                    [&](LoadOperation&) { QUNIT_IS_TRUE(false); },
                    [&](ApplyOperation&) { QUNIT_IS_TRUE(false); },
                    [&](FilterOperation&) { QUNIT_IS_TRUE(false); },
                    [&](HoistOperation& hoistOperation) {
                        QUNIT_IS_EQUAL("\"homeworkAverage\"", hoistOperation.hoistTarget)

                        QUNIT_IS_EQUAL("B", hoistOperation.inputTable.contents);

                        QUNIT_IS_EQUAL("student", hoistOperation.inputTableColumnName.contents);

                        QUNIT_IS_TRUE(hoistOperation.retain->isAll());
                    },
                    [&](BinaryOperation&) { QUNIT_IS_TRUE(false); });
            },
            [&](StoreOperation&) { QUNIT_IS_TRUE(false); });

    parseTree.statements->operator[](statementNumber++)
        ->match(
            [&](TableAssignment& assignment) {
                QUNIT_IS_EQUAL("D", assignment.tableName.contents)

                QUNIT_IS_EQUAL(2, assignment.columnNames->size());

                QUNIT_IS_EQUAL("student", assignment.columnNames->operator[](0).contents);
                QUNIT_IS_EQUAL("isExamGreater", assignment.columnNames->operator[](1).contents)

                assignment.value->match(
                    [&](LoadOperation&) { QUNIT_IS_TRUE(false); },
                    [&](ApplyOperation&) { QUNIT_IS_TRUE(false); },
                    [&](FilterOperation&) { QUNIT_IS_TRUE(false); },
                    [&](HoistOperation&) { QUNIT_IS_TRUE(false); },
                    [&](BinaryOperation& binOp) {
                        binOp.execute([&](GreaterThanOp gt) {
                            QUNIT_IS_EQUAL("C", gt.lhsTableName.contents);
                            QUNIT_IS_EQUAL("examAverage", gt.lhsColumnName.contents);

                            QUNIT_IS_EQUAL("C", gt.rhsTableName.contents);
                            QUNIT_IS_EQUAL("hwAverage", gt.rhsColumnName.contents);
                        });
                    });
            },
            [&](StoreOperation&) { QUNIT_IS_TRUE(false); });

    parseTree.statements->operator[](statementNumber++)
        ->match(
            [&](TableAssignment& assignment) {
                QUNIT_IS_EQUAL("E", assignment.tableName.contents)

                QUNIT_IS_EQUAL(1, assignment.columnNames->size());

                QUNIT_IS_EQUAL("student", assignment.columnNames->operator[](0).contents);

                assignment.value->match(
                    [&](LoadOperation) { QUNIT_IS_TRUE(false); },
                    [&](ApplyOperation applyOperation) { QUNIT_IS_TRUE(false); },
                    [&](FilterOperation filterOperation) {
                        QUNIT_IS_EQUAL("D", filterOperation.inputTableName.contents)
                        QUNIT_IS_EQUAL("isExamGreater", filterOperation.filterColumnName.contents)

                        filterOperation.retain->match(
                            [&](RetainAllClause) { QUNIT_IS_TRUE(false); },
                            [&](RetainExplicitClause exp) {
                                QUNIT_IS_EQUAL(1, exp.columns->size());
                                QUNIT_IS_EQUAL("student", exp.columns->operator[](0).contents)
                            },
                            [&](RetainNoneClause) { QUNIT_IS_TRUE(false); });
                    },
                    [&](HoistOperation&) { QUNIT_IS_TRUE(false); },
                    [&](BinaryOperation&) { QUNIT_IS_TRUE(false); });
            },
            [&](StoreOperation&) { QUNIT_IS_TRUE(false); });

    parseTree.statements->operator[](statementNumber++)
        ->match(
            [&](TableAssignment& assignment) {
                QUNIT_IS_EQUAL("F", assignment.tableName.contents)

                QUNIT_IS_EQUAL(1, assignment.columnNames->size());

                QUNIT_IS_EQUAL("name", assignment.columnNames->operator[](0).contents);

                assignment.value->match(
                    [&](LoadOperation&) { QUNIT_IS_TRUE(false); },
                    [&](ApplyOperation& applyOperation) {
                        QUNIT_IS_EQUAL(ApplyOperationType::method, applyOperation.applyType);

                        QUNIT_IS_EQUAL("\"getName\"", applyOperation.applyTarget)

                        QUNIT_IS_EQUAL("E", applyOperation.inputTable.contents)

                        QUNIT_IS_EQUAL(1, applyOperation.inputTableColumnNames->size());

                        QUNIT_IS_EQUAL(
                            "student",
                            applyOperation.inputTableColumnNames->operator[](0).contents);

                        QUNIT_IS_TRUE(applyOperation.retain->isNone());
                    },
                    [&](FilterOperation&) { QUNIT_IS_TRUE(false); },
                    [&](HoistOperation&) { QUNIT_IS_TRUE(false); },
                    [&](BinaryOperation&) { QUNIT_IS_TRUE(false); });
            },
            [&](StoreOperation&) { QUNIT_IS_TRUE(false); });

    parseTree.statements->operator[](statementNumber++)
        ->match([&](TableAssignment& assignment) { QUNIT_IS_TRUE(false); },
                [&](StoreOperation& store) {
                    QUNIT_IS_EQUAL("F", store.outputTable.contents);
                    QUNIT_IS_EQUAL("\"(databaseName, outputSetName)\"", store.destination)
                });
}

void testParseTcap1(UnitTest& qunit) {
    string program =
        "@exec \"exec1\"\n"
        "A(student) = load \"(databaseName, inputSetName)\"\n"
        "B(student, examAverage) = apply func \"avgExams\" to A[student] retain all\n"
        "C(student, examAverage, hwAverage) = hoist \"homeworkAverage\" from B[student] retain "
        "all\n"
        "D(student, isExamGreater) = C[examAverage] > C[hwAverage] retain student\n"
        "E(student) = filter D by isExamGreater retain student\n"
        "F(name) = apply method \"getName\" to E[student] retain none"
        "store F[name] \"(databaseName, outputSetName)\"";

    shared_ptr<SafeResult<TranslationUnit>> parseTreeResult = parseTcap(program);

    parseTreeResult->apply([&](TranslationUnit parseTree) { testParseTcap1Help(qunit, parseTree); },
                           [&](string errorMsg) { QUNIT_IS_TRUE(false); });
}
}

#endif  // PDB_TCAPPARSERTESTS_PARSETCAPTESTS_H
