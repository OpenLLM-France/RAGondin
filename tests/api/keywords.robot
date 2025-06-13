*** Settings ***
Library     String
Library     Collections
Library    RequestsLibrary

*** Variables ***
${BASE}           
${PORT}           
${test_file_1}    test_file_1.pdf
${test_file_2}    test_file_2.pdf
${test_part_1}    test
${test_part_2}    test2
${BASE_URL}       ${BASE}:${PORT}

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
    [Arguments]    ${task_id}    ${expected_status}=200
    ${response}=    GET    ${BASE_URL}/indexer/task/${task_id}    expected_status=${expected_status}
    RETURN    ${response.json()}

Get Extract
    [Arguments]    ${extract_id}    ${expected_status}=200
    ${response}=    GET    ${BASE_URL}/extract/${extract_id}    expected_status=${expected_status}
    RETURN    ${response.json()}

Index File Non Blocking
    [Arguments]    ${file_path}    ${id}    ${part}    ${expected_status}=201
    ${file}=    Get File For Streaming Upload    ${file_path}
    ${files}=    Create Dictionary    file=${file}
    ${response}=    POST
    ...    ${BASE_URL}/indexer/partition/${part}/file/${id}
    ...    files=${files}
    ...    expected_status=${expected_status}
    ${response}=    Set Variable    ${response.json()}
    Should Match Regexp    ${response}[task_status_url]    ${BASE_URL}/indexer/task/[a-fA-F0-9]{48}
    RETURN    ${response}

Index File
    [Arguments]    ${file_path}    ${id}    ${part}    ${expected_status}=201
    ${file}=    Get File For Streaming Upload    ${file_path}
    ${files}=    Create Dictionary    file    ${file}
    ${response}=    POST
    ...    ${BASE_URL}/indexer/partition/${part}/file/${id}
    ...    files=${files}
    ...    expected_status=${expected_status}
    ${response}=    Set Variable    ${response.json()}
    Should Match Regexp    ${response}[task_status_url]    ${BASE_URL}/indexer/task/[a-fA-F0-9]{48}
    ${task_id}=    Fetch From Right    ${response}[task_status_url]    /
    Sleep    1
    FOR    ${i}    IN RANGE    0    60    # 60 seconds
        ${response}=    Get Task Status    ${task_id}
        IF    '${response}[task_state]' == 'COMPLETED'    BREAK
        Sleep    1
        IF    ${i} == 59
            Log    Task '${task_id}' is still running after 60 seconds.
            Log    ${response}
            Fail    Task '${task_id}' is still running after 60 seconds.
        END
    END

Check File Exists
    [Arguments]    ${id}    ${part}    ${expected_status}=200
    ${response}=    GET    ${BASE_URL}/partition/check-file/${part}/file/${id}    expected_status=${expected_status}
    IF    '${expected_status}' == '200'
        Should Be Equal As Strings    ${response.json()}    File '${id}' exists in partition '${part}'.
    END
    IF    '${expected_status}' == '404'
        Should Be Equal As Strings    ${response.json()}[detail]    File '${id}' not found in partition '${part}'.
    END

Patch File
    [Arguments]    ${id}    ${part}    ${metadata}    ${expected_status}=200
    ${form_data}=    Create Dictionary    metadata=${metadata}
    ${response}=    PATCH
    ...    ${BASE_URL}/indexer/partition/${part}/file/${id}
    ...    data=${form_data}
    ...    expected_status=${expected_status}
    IF    '${expected_status}' == '200'
        Should Be Equal As Strings    ${response.json()}[message]    Metadata for file '${id}' successfully updated.
    END

Delete File
    [Arguments]    ${id}    ${part}=${test_part_1}    ${expected_status}=204
    ${response}=    DELETE    ${BASE_URL}/indexer/partition/${part}/file/${id}    expected_status=${expected_status}
    RETURN    None

Delete Partition
    [Arguments]    ${part}    ${expected_status}=204
    ${response}=    DELETE    ${BASE_URL}/partition/${part}    expected_status=${expected_status}
    RETURN    None

Get File Metadata
    [Arguments]    ${id}    ${part}    &{expected_metadata_and_status}
    ${expected_status}=    Get From Dictionary    ${expected_metadata_and_status}    expected_status    200
    ${expected_metadata}=    Remove From Dictionary    ${expected_metadata_and_status}    expected_status
    ${response}=    GET    ${BASE_URL}/partition/${part}/file/${id}    expected_status=${expected_status}
    ${json_response}=    Set Variable    ${response.json()}
    IF    '${expected_status}' == '404'
        Should Be Equal As Strings    ${json_response}[detail]    File '${id}' not found in partition '${part}'.
    END
    IF    ${expected_metadata}    # Check if expected_metadata is not empty
        FOR    ${key}    IN    @{expected_metadata.keys()}
            Should Be Equal    ${json_response['metadata']['${key}']}    ${expected_metadata['${key}']}
        END
    ELSE
        Log    No expected metadata to validate.
    END

Get Models
    ${response}=    GET    ${BASE_URL}/v1/models
    Log To Console    ${response}