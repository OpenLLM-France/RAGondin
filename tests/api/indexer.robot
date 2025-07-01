*** Settings ***
Resource    keywords.robot

*** Test Cases ***
Add File and Delete it
    Index File    ${CURDIR}/${test_file_1}    0    test
    Delete File    0    test
    [Teardown]    Clean Up Test    test

Get Non Existent Task Status
    ${response}=    Get Task Status    82891771158d68c1eacb9d1f151391007f68c96901000000    404
    Should Be Equal As Strings
    ...    ${response}[detail]
    ...    Task '82891771158d68c1eacb9d1f151391007f68c96901000000' not found.

Get Invalid Task Id Status (-1)
    ${response}=    Get Task Status    -1    404
    Should Be Equal As Strings    ${response}[detail]    Task '-1' not found.

Get Invalid Task Id Status (123)
    ${response}=    Get Task Status    123    404
    Should Be Equal As Strings    ${response}[detail]    Task '123' not found.

Add File and Check Metadata
    &{file_metadata}=    Create Dictionary    filename=${test_file_1}    partition=test    file_id=0
    Index File    ${CURDIR}/${test_file_1}    0    test
    ${response}=    Get File Metadata    0    test    &{file_metadata}
    [Teardown]    Clean Up Test    test

Add File and Patch it with new metadata
    Index File    ${CURDIR}/${test_file_2}    0    test
    ${metadata}=    Create Dictionary    title=Test Title    author=Test Author
    ${metadata}=    Evaluate    json.dumps(${metadata})    json
    Patch File    0    test    ${metadata}
    &{file_metadata}=    Create Dictionary    title=Test Title    author=Test Author
    ${response}=    Get File Metadata    0    test    &{file_metadata}
    [Teardown]    Clean Up Test    test

Get Non Existent File
    Get File Metadata    id=0    part=test    expected_status=404

Get Invalid File Id (-1)
    Get File Metadata    id=-1    part=test    expected_status=404

Get Invalid File Id (`"&é\'-!")
    Get File Metadata    id="&é\'-!    part=test    expected_status=404

Test Indexer Overload
    ${task_ids}=    Create List
    FOR    ${i}    IN RANGE    0    100
        ${response}=    Index File Non Blocking    ${CURDIR}/${test_file_1}    ${i}    test
        ${task_url}=    Set Variable    ${response}[task_status_url]
        ${task_id}=    Fetch From Right    ${task_url}    /
        Append To List    ${task_ids}    ${task_id}
    END
    Sleep    1
    FOR    ${i}    IN RANGE    0    1200
        FOR    ${task_id}    IN    @{task_ids}
            ${response}=    Get Task Status    ${task_id}
            ${task_state}=    Set Variable    ${response}[task_state]
            IF    '${task_state}' == 'FAILED'
                Fail    Task '${task_id}' failed.
            ELSE IF    '${task_state}' == 'COMPLETED'
                Remove Values From List    ${task_ids}    ${task_id}
            END
        END
        Sleep    1
        ${list_length}=    Get Length    ${task_ids}
        IF    ${list_length} > 0
            Log    ${task_ids}
        END
        # Exit early if all tasks completed
        ${list_length}=    Get Length    ${task_ids}
        IF    ${list_length} == 0
            BREAK
        END
    END
    IF   ${list_length} > 0
        Log    Some tasks are still running after 1200 seconds: ${task_ids}
        Fail    Some tasks are still running after 1200 seconds.
    END
    [Teardown]    Clean Up Test    test
