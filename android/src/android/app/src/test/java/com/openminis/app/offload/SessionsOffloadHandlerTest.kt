package com.openminis.app.offload

import com.openminis.app.data.repository.ChatRepository
import com.openminis.app.sandbox.offload.SessionsOffloadHandler
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** JVM-level tests for the minis-sessions-cli argument surface (ADR: send). */
class SessionsOffloadHandlerTest {

    @Test
    fun `send without message is an invalid-args envelope`() {
        // The send validation path returns before the repository is touched,
        // so an unresolved repository stub is safe here (JVM tests cannot
        // build a Room-backed ChatRepository).
        // The send validation path returns before the repository is touched,
        // so a lazily-throwing repository stub is safe here (JVM tests cannot
        // build a Room-backed ChatRepository; the dao is never dereferenced
        // on this path).
        val neverDao = java.lang.reflect.Proxy.newProxyInstance(
            ChatRepository::class.java.classLoader,
            arrayOf(com.openminis.app.data.db.ChatDao::class.java),
        ) { _, _, _ -> error("dao must not be touched on the validation path") }
        val handler = SessionsOffloadHandler(
            repo = ChatRepository(neverDao as com.openminis.app.data.db.ChatDao),
        )
        val result = handler.handle(
            com.openminis.app.sandbox.NativeOffloadRequest(
                pid = 1,
                argv = listOf("minis-sessions-cli", "send"),
                env = emptyMap(),
                cwd = "/",
            ),
        )
        assertEquals(2 /* EXIT_INVALID_ARGS */, result.exitCode)
        val body = result.output
        assertTrue(body.contains("INVALID_ARGS"))
        assertTrue(body.contains("--message"))
    }
}
