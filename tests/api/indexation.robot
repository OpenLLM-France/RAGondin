*** Settings ***
Resource    keywords.robot

*** Test Cases ***
Add File and Delete it
    Index File    ${CURDIR}/${test_file_1}    0    test
    Delete File    0    test
    [Teardown]    Clean Up Test    test