*** Settings ***
Resource    keywords.robot

*** Test Cases ***
Add files to two partitions and search each partition
    Index File    ${CURDIR}/${test_file_1}    0    test
    Index File    ${CURDIR}/${test_file_1}    1    test2
    Check File Exists    0    test
    Check File Exists    1    test    404
    Check File Exists    1    test2
    Check File Exists    0    test2    404
    [Teardown]    Clean Up Test    test    test2