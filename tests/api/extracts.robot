*** Settings ***
Resource    keywords.robot

*** Test Cases ***
Get Non Existent Extract
    ${response}=    Get Task Status    82891771158d68c1eacb9d1f151391007f68c96901000000    404
    Should Be Equal As Strings
    ...    ${response}[detail]
    ...    Task '82891771158d68c1eacb9d1f151391007f68c96901000000' not found.

Get Invalid Extract Id (-1)
    ${response}=    Get Task Status    -1    404
    Should Be Equal As Strings    ${response}[detail]    Task '-1' not found.

Get Invalid Extract Id (123)
    ${response}=    Get Task Status    123    404
    Should Be Equal As Strings    ${response}[detail]    Task '123' not found.