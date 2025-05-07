*** Settings ***
Library    RequestsLibrary
Library    String
Library    Collections

*** Variables
${BASE}  
${PORT}  
${test_file_1}    
${test_file_2}    
${BASE_URL}  ${BASE}:${PORT}


*** Keywords ***
Clean Up Test
    [Arguments]    @{part}
    ${allowed_status}=    Create List    204    404
    FOR    ${partition}    IN    @{part}
        ${response}=    DELETE    ${BASE_URL}/partition/${partition}    expected_status=any
        ${status_code}=    Convert To String    ${response.status_code}
        List Should Contain Value    ${allowed_status}    ${status_code}
    END


Get Task Status
    [Arguments]  ${task_id}  ${expected_status}=200
    ${response}=  GET  ${BASE_URL}/indexer/task/${task_id}  expected_status=${expected_status}
    RETURN  ${response.json()}

Get Extract
    [Arguments]    ${extract_id}    ${expected_status}=200
    ${response}=    GET    ${BASE_URL}/extract/${extract_id}    expected_status=${expected_status}
    RETURN    ${response.json()}

Index File
    [Arguments]    ${file_path}    ${id}    ${part}    ${expected_status}=201
    ${file}=    Get File For Streaming Upload    ${file_path}
    ${files}=    Create Dictionary    file    ${file}
    ${response}=    POST    ${BASE_URL}/indexer/partition/${part}/file/${id}    files=${files}    expected_status=${expected_status}
    ${response}=    Set Variable    ${response.json()}
    Should Match Regexp    ${response}[task_status_url]    ${BASE_URL}/indexer/task/[a-fA-F0-9]{48}
    ${task_id}=    Fetch From Right    ${response}[task_status_url]    /
    Sleep    1
    FOR  ${i}  IN RANGE  0  30  # 30 seconds
        ${response}=    Get Task Status    ${task_id}
        Run Keyword If    '${response}[task_state]' == 'FINISHED'    Exit For Loop
        Sleep    1
    END

Check File Exists 
    [Arguments]  ${id}  ${part}  ${expected_status}=200
    ${response}=    GET    ${BASE_URL}/partition/check-file/${part}/file/${id}    expected_status=${expected_status}
    Run Keyword If    '${expected_status}' == '200'    Should Be Equal As Strings    ${response.json()}    File '${id}' exists in partition '${part}'.
    Run Keyword If    '${expected_status}' == '404'    Should Be Equal As Strings    ${response.json()}[detail]    File '${id}' not found in partition '${part}'.

Patch File
    [Arguments]    ${id}    ${part}    ${metadata}    ${expected_status}=200
    ${form_data}=    Create Dictionary    metadata=${metadata}
    ${response}=    PATCH    ${BASE_URL}/indexer/partition/${part}/file/${id}    data=${form_data}    expected_status=${expected_status}
    Run Keyword If    '${expected_status}' == '200'    Should Be Equal As Strings    ${response.json()}[message]    Metadata for file '${id}' successfully updated.


Delete File
    [Arguments]  ${id}    ${part}=test    ${expected_status}=204
    ${response}=    DELETE    ${BASE_URL}/indexer/partition/${part}/file/${id}    expected_status=${expected_status}
    RETURN    None


Delete Partition
    [Arguments]  ${part}  ${expected_status}=204
    ${response}=    DELETE    ${BASE_URL}/partition/${part}    expected_status=${expected_status}
    RETURN    None

Get File Metadata
    [Arguments]    ${id}    ${part}    ${expected_status}=200    &{expected_metadata}
    ${is_empty}=    Evaluate    not bool(${expected_metadata})
    Run Keyword If    ${is_empty}    Set Variable    &{expected_metadata}    Create Dictionary
    ${response}=    GET    ${BASE_URL}/partition/${part}/file/${id}    expected_status=${expected_status}
    ${json_response}=    Set Variable    ${response.json()}
    Log to Console    ${json_response}
    Log to Console    ${json_response['metadata']}
    Run Keyword If    '${expected_status}' == '404'    Should Be Equal As Strings    ${json_response}[detail]    File '${id}' not found in partition '${part}'.
    FOR    ${key}    IN    @{expected_metadata.keys()}
        Should Be Equal    ${json_response['metadata']['${key}']}    ${expected_metadata['${key}']}
    END

*** Test Cases ***
Health Check
    ${response}=    GET    ${BASE_URL}/health_check    expected_status=200
    Should Be Equal As Strings    ${response.json()}    RAG API is up.
    [Teardown]

Add File and Check Metadata
    &{file_metadata}=    Create Dictionary    filename=eng.pdf    partition=test    file_id=0
    Index File    ${test_file_1}    0    test
    ${response}=    Get File Metadata    0    test    &{file_metadata}
    [Teardown]    Clean Up Test    test

Add File and Patch it with new metadata
    Index File    ${test_file_2}    0    test
    ${metadata}=    Create Dictionary    title=Test Title    author=Test Author
    ${metadata}=    Evaluate    json.dumps(${metadata})    json
    Patch File    0    test    ${metadata}
    &{file_metadata}=    Create Dictionary    title=Test Title    author=Test Author
    ${response}=    Get File Metadata    0    test    &{file_metadata}
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


# Shoud Fail tests 
Get Non Existent Task Status
    ${response}=  Get Task Status  82891771158d68c1eacb9d1f151391007f68c96901000000  404
    Should Be Equal As Strings  ${response}[detail]  Task '82891771158d68c1eacb9d1f151391007f68c96901000000' not found.
Get Invalid Task Id Status (-1)
    ${response}=  Get Task Status  -1  404
    Should Be Equal As Strings  ${response}[detail]  Task '-1' not found.
Get Invalid Task Id Status (123)
    ${response}=  Get Task Status  123  404
    Should Be Equal As Strings  ${response}[detail]  Task '123' not found.

Get Non Existent Extract
    ${response}=  Get Task Status  82891771158d68c1eacb9d1f151391007f68c96901000000  404
    Should Be Equal As Strings  ${response}[detail]  Task '82891771158d68c1eacb9d1f151391007f68c96901000000' not found.
Get Invalid Extract Id (-1)
    ${response}=  Get Task Status  -1  404
    Should Be Equal As Strings  ${response}[detail]  Task '-1' not found.
Get Invalid Extract Id (123)
    ${response}=  Get Task Status  123  404
    Should Be Equal As Strings  ${response}[detail]  Task '123' not found.

Get Non Existent File
    Get File Metadata    0    test    404
Get Invalid File Id (-1)
    Get File Metadata    -1    test    404
Get Invalid File Id (`"&é\'-!/")
    Get File Metadata    "&é\'-!/    test    404