from source import run

# test_caseX = (<.pnml file name> , <cost in task5>, <print detail test infomation>)

test_case0 = ("report.pnml", [1,2,3], True)

test_case1 = ("medium_deadlock.pnml", [1, 2, 3, 4, 5, 6, 7, 8], True)

test_case2 = ("medium_no_deadlock.pnml", [5, 1, 1, 1, 1, 1, 1, 2], True)

# Change this line to run  different testcase
run(test_case0)
