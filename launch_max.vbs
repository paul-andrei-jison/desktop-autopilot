Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "C:\Users\Paul Jison\Documents\AI Practice\DesktopAutopilotAI\desktop-autopilot"

Function OllamaUp()
    OllamaUp = False
    On Error Resume Next
    Set http = CreateObject("WinHttp.WinHttpRequest.5.1")
    http.Open "GET", "http://127.0.0.1:11434/", False
    http.SetTimeouts 1000, 1000, 1000, 1000
    http.Send
    If Err.Number = 0 Then OllamaUp = True
End Function

If Not OllamaUp() Then
    WshShell.Run """" & "C:\Users\Paul Jison\AppData\Local\AMD\AI_Bundle\Ollama\ollama.EXE" & """ serve", 0, False
    For i = 1 To 30
        WScript.Sleep 1000
        If OllamaUp() Then Exit For
    Next
End If

WshShell.Run """" & "C:\Users\Paul Jison\Documents\AI Practice\DesktopAutopilotAI\desktop-autopilot\venv\Scripts\pythonw.exe" & """ brain.py", 0, False
