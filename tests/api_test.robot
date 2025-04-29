*** Settings ***
Library  RequestsLibrary
Library  String

*** Variables
${BASE}  http://163.114.159.68
${PORT}  8089
${test_file_1}    ./info-paul.pdf
${test_file_2}    ./Projet_2.pdf
${BASE_URL}  ${BASE}:${PORT}


*** Keywords ***
Clean Up Test
    [Arguments]  @{partition}
    FOR    ${part}    IN    @{partition}
        ${response}=    DELETE    ${BASE_URL}/partition/${part}    expected_status=any
    END
    RETURN    None


Get Task Status
    [Arguments]  ${task_id}  ${expected_status}=200
    ${response}=  GET  ${BASE_URL}/indexer/task/${task_id}  expected_status=${expected_status}
    RETURN  ${response.json()}

Get Extract
    [Arguments]    ${extract_id}    ${partition}    ${expected_status}=200
    ${response}=    GET    ${BASE_URL}/extract/${extract_id}    expected_status=${expected_status}
    RETURN    ${response.json()}

Index File
    [Arguments]    ${file_path}    ${file_id}    ${partition}    ${expected_status}=201
    ${file}=    Get File For Streaming Upload    ${file_path}
    ${files}=    Create Dictionary    file    ${file}
    ${response}=    POST    ${BASE_URL}/indexer/partition/${partition}/file/${file_id}    files=${files}    expected_status=${expected_status}
    ${response}=    Set Variable    ${response.json()}
    Should Match Regexp    ${response}[task_status_url]    ${BASE_URL}/indexer/task/[a-fA-F0-9]{48}
    ${task_id}=    Fetch From Right    ${response}[task_status_url]    /
    Sleep    1
    FOR  ${i}  IN RANGE  0  30  # 30 seconds
        ${response}=    Get Task Status    ${task_id}
        Run Keyword If    '${response}[task_state]' == 'FINISHED'    Exit For Loop
        Sleep    1
    END

Get File Metadata
    [Arguments]  ${file_id}  ${partition}  ${expected_status}=200
    ${response}=    GET    ${BASE_URL}/partition/${partition}/file/${file_id}    expected_status=${expected_status}
    RETURN    ${response.json()}[metadata]
    

Check File Exists 
    [Arguments]  ${file_id}  ${partition}  ${expected_status}=200
    ${response}=    GET    ${BASE_URL}/partition/check-file/${partition}/file/${file_id}    expected_status=${expected_status}
    Run Keyword If    '${expected_status}' == '200'    Should Be Equal As Strings    ${response.json()}    File '${file_id}' exists in partition '${partition}'.
    Run Keyword If    '${expected_status}' == '404'    Should Be Equal As Strings    ${response.json()}[detail]    File '${file_id}' not found in partition '${partition}'.

Patch File
    [Arguments]    ${file_id}    ${partition}    ${metadata}    ${expected_status}=200
    ${form_data}=    Create Dictionary    metadata=${metadata}
    ${response}=    PATCH    ${BASE_URL}/indexer/partition/${partition}/file/${file_id}    data=${form_data}    expected_status=${expected_status}
    Run Keyword If    '${expected_status}' == '200'    Should Be Equal As Strings    ${response.json()}[message]    Metadata for file '${file_id}' successfully updated.


Delete File
    [Arguments]  ${file_id}    ${partition}=test    ${expected_status}=204
    ${response}=    DELETE    ${BASE_URL}/indexer/partition/${partition}/file/${file_id}    expected_status=${expected_status}
    RETURN    None


Delete Partition
    [Arguments]  ${partition}  ${expected_status}=204
    ${response}=    DELETE    ${BASE_URL}/partition/${partition}    expected_status=${expected_status}
    RETURN    None

*** Test Cases ***
Health Check
    ${response}=    GET    ${BASE_URL}/health_check    expected_status=200
    Should Be Equal As Strings    ${response.json()}    RAG API is up.
    [Teardown]

Add File and Patch it with new metadata
    Index File    ${test_file_2}    0    test
    ${metadata}=    Create Dictionary    title=Test Title    author=Test Author
    ${metadata}=    Evaluate    json.dumps(${metadata})    json
    Log To Console    ${metadata}
    Patch File    0    test    ${metadata}
    ${response}=    Get File Metadata    0    test
    Log To Console    ${response}
    Should Be Equal As Strings    ${response}[title]    Test Title
    Should Be Equal As Strings    ${response}[author]    Test Author
    [Teardown]    Clean Up Test    test


Add File and Delete it
    Index File    ${test_file_1}    0    test
    Delete File    0    test
    [Teardown]    Clean Up Test    test

Add files to two partitions and search each partition
    Index File    ${test_file_1}    0    test
    Index File    ${test_file_1}    1    test2
    Check File Exists    0    test
    Check File Exists    1    test    404
    Check File Exists    1    test2
    Check File Exists    0    test2    404
    [Teardown]    Clean Up Test    test    test2



# Get Non Existent Task Status
#     ${response}=  Get Task Status  82891771158d68c1eacb9d1f151391007f68c96901000000  404
#     Should Be Equal As Strings  ${response}[detail]  Task '82891771158d68c1eacb9d1f151391007f68c96901000000' not found.
# Get Invalid Task Id Status (-1)
#     ${response}=  Get Task Status  -1  404
#     Should Be Equal As Strings  ${response.json()}[detail]  '-1' is not a valid task id.
# Get Invalid Task Id Status (123)
#     ${response}=  Get Task Status  123  400
#     Should Be Equal As Strings  ${response.json()}[detail]  '123' is not a valid task id.

# Get Non Existent Extract
#     ${response}=  Get Task Status  82891771158d68c1eacb9d1f151391007f68c96901000000  404
#     Should Be Equal As Strings  ${response}[detail]  Task '82891771158d68c1eacb9d1f151391007f68c96901000000' not found.
# Get Invalid Extract Id (-1)
#     ${response}=  Get Task Status  -1  400
#     Should Be Equal As Strings  ${response.json()}[detail]  '-1' is not a valid task id.
# Get Invalid Extract Id (123)
#     ${response}=  Get Task Status  123  400
#     Should Be Equal As Strings  ${response.json()}[detail]  '123' is not a valid task id.
