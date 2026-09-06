// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useState } from 'react'
import {
    Badge,
    Button,
    Flex,
    FormField,
    Stack,
    Text,
    TextInput
} from '@nvidia/foundations-react-core'
import getErrorString from '@/shared/utils/get-error-string'
import { InlineErrorBanner } from '@/ui/components/InlineErrorBanner'
import { useConnectionStore } from '@/ui/stores/connection.store'

interface AuthStatus {
    initialized: boolean
    administrator: boolean
    clusterId: string
    revision: number
}

interface ClientStatus {
    id: string
    label: string
    revoked: boolean
}

const CREDENTIAL_PATTERN = /^engine\.[A-Za-z0-9._-]+\.api_key$/

export default function AuthenticationSettings() {
    const connected = useConnectionStore(state => state.connected)
    const clusterId = useConnectionStore(state => state.clusterId)
    const [auth, setAuth] = useState<AuthStatus | null>(null)
    const [clients, setClients] = useState<ClientStatus[]>([])
    const [clientLabel, setClientLabel] = useState('')
    const [newToken, setNewToken] = useState<string | null>(null)
    const [credential, setCredential] = useState('engine.example.api_key')
    const [credentialValue, setCredentialValue] = useState('')
    const [credentialConfigured, setCredentialConfigured] = useState(false)
    const [credentialSource, setCredentialSource] = useState('missing')
    const [busy, setBusy] = useState<string | null>(null)
    const [error, setError] = useState<string | null>(null)

    const refreshAuth = useCallback(async () => {
        const status = await window.pairApi.auth.status()
        setAuth(status)
        if (status.initialized && status.administrator) {
            const result = await window.pairApi.auth.listClients()
            setClients(result.clients)
        } else {
            setClients([])
        }
    }, [])

    const inspectCredential = useCallback(async () => {
        const ref = credential.trim()
        if (!CREDENTIAL_PATTERN.test(ref)) {
            setCredentialConfigured(false)
            setCredentialSource('missing')
            return
        }
        const status = await window.pairApi.credentials.status(ref)
        setCredentialConfigured(status.configured)
        setCredentialSource(status.source)
    }, [credential])

    useEffect(() => {
        if (!connected) return
        setError(null)
        void refreshAuth().catch(err => setError(getErrorString(err)))
    }, [connected, clusterId, refreshAuth])

    const run = useCallback(async (name: string, action: () => Promise<void>) => {
        setBusy(name)
        setError(null)
        try {
            await action()
        } catch (err) {
            setError(getErrorString(err))
        } finally {
            setBusy(null)
        }
    }, [])

    const bootstrap = () =>
        run('bootstrap', async () => {
            await window.pairApi.auth.bootstrapOwner()
            await refreshAuth()
        })

    const createClient = () => {
        const label = clientLabel.trim()
        if (!label) return
        return run('create-client', async () => {
            const created = await window.pairApi.auth.createClient(label)
            setNewToken(created.token)
            setClientLabel('')
            await refreshAuth()
        })
    }

    const revokeClient = (id: string) =>
        run(`revoke-${id}`, async () => {
            await window.pairApi.auth.revokeClient(id)
            await refreshAuth()
        })

    const saveCredential = () => {
        const ref = credential.trim()
        if (!CREDENTIAL_PATTERN.test(ref) || !credentialValue) return
        return run('set-credential', async () => {
            const status = await window.pairApi.credentials.set(ref, credentialValue)
            setCredentialValue('')
            setCredentialConfigured(status.configured)
            setCredentialSource(status.source)
        })
    }

    const clearCredential = () => {
        const ref = credential.trim()
        if (!CREDENTIAL_PATTERN.test(ref)) return
        return run('clear-credential', async () => {
            const status = await window.pairApi.credentials.clear(ref)
            setCredentialValue('')
            setCredentialConfigured(status.configured)
            setCredentialSource(status.source)
        })
    }

    return (
        <Stack gap="6" className="relative py-8 px-3 w-full">
            {error && <InlineErrorBanner severity="error" message={error} />}

            <div className="settings-card settings-card-stacked pair-paper p-4">
                <Stack gap="4">
                    <Flex justify="between" align="center" gap="3" wrap="wrap">
                        <Stack gap="1">
                            <Text kind="body/semibold/md">PAIR client access</Text>
                            <Text kind="body/regular/sm" className="text-subtle-color">
                                Client tokens are valid across this cluster. Only the founding node
                                can manage them.
                            </Text>
                        </Stack>
                        <Badge color={auth?.administrator ? 'green' : 'gray'} kind="solid">
                            {auth?.administrator
                                ? 'Administrator'
                                : auth?.initialized
                                  ? 'Managed by another node'
                                  : 'Not initialized'}
                        </Badge>
                    </Flex>

                    {!auth?.initialized && (
                        <Button
                            kind="primary"
                            color="brand"
                            size="small"
                            disabled={!connected || !clusterId || busy !== null}
                            onClick={() => void bootstrap()}
                        >
                            Initialize authentication
                        </Button>
                    )}

                    {auth?.administrator && (
                        <>
                            <Flex align="end" gap="2" wrap="wrap">
                                <FormField slotLabel="Client label" className="flex-1">
                                    <TextInput
                                        value={clientLabel}
                                        onValueChange={setClientLabel}
                                        placeholder="hermes-agent"
                                        maxLength={80}
                                        onKeyDown={event => {
                                            if (event.key === 'Enter') void createClient()
                                        }}
                                    />
                                </FormField>
                                <Button
                                    kind="primary"
                                    color="brand"
                                    size="small"
                                    disabled={!clientLabel.trim() || busy !== null}
                                    onClick={() => void createClient()}
                                >
                                    Create token
                                </Button>
                            </Flex>

                            {newToken && (
                                <Stack
                                    gap="2"
                                    className="p-3 border border-warning-color rounded-sm"
                                >
                                    <Text kind="body/semibold/sm">Save this token now</Text>
                                    <Text kind="body/regular/sm" className="text-subtle-color">
                                        PAIR cannot show it again after you dismiss it or leave this
                                        page.
                                    </Text>
                                    <code className="break-all select-all">{newToken}</code>
                                    <Button
                                        kind="secondary"
                                        size="small"
                                        onClick={() => setNewToken(null)}
                                    >
                                        I saved it
                                    </Button>
                                </Stack>
                            )}

                            <Stack gap="2">
                                {clients.length === 0 ? (
                                    <Text kind="body/regular/sm" className="text-subtle-color">
                                        No clients yet.
                                    </Text>
                                ) : (
                                    clients.map(client => (
                                        <Flex
                                            key={client.id}
                                            justify="between"
                                            align="center"
                                            gap="3"
                                        >
                                            <Stack gap="0">
                                                <Text kind="body/semibold/sm">{client.label}</Text>
                                                <Text
                                                    kind="body/regular/sm"
                                                    className="text-subtle-color"
                                                >
                                                    {client.id}
                                                </Text>
                                            </Stack>
                                            {client.revoked ? (
                                                <Badge color="gray" kind="solid">
                                                    Revoked
                                                </Badge>
                                            ) : (
                                                <Button
                                                    kind="tertiary"
                                                    color="danger"
                                                    size="small"
                                                    disabled={busy !== null}
                                                    onClick={() => void revokeClient(client.id)}
                                                >
                                                    Revoke
                                                </Button>
                                            )}
                                        </Flex>
                                    ))
                                )}
                            </Stack>
                        </>
                    )}
                </Stack>
            </div>

            <div className="settings-card settings-card-stacked pair-paper p-4">
                <Stack gap="4">
                    <Stack gap="1">
                        <Text kind="body/semibold/md">Local engine credential</Text>
                        <Text kind="body/regular/sm" className="text-subtle-color">
                            Stored in this node&apos;s native credential store. The saved value is
                            never returned.
                        </Text>
                    </Stack>
                    <FormField slotLabel="Credential reference">
                        <TextInput
                            value={credential}
                            onValueChange={setCredential}
                            placeholder="engine.example.api_key"
                        />
                    </FormField>
                    <Flex align="center" gap="2" wrap="wrap">
                        <Button
                            kind="secondary"
                            size="small"
                            disabled={
                                !connected ||
                                busy !== null ||
                                !CREDENTIAL_PATTERN.test(credential.trim())
                            }
                            onClick={() => void run('inspect-credential', inspectCredential)}
                        >
                            Inspect presence
                        </Button>
                        <Badge color={credentialConfigured ? 'green' : 'gray'} kind="solid">
                            {credentialConfigured
                                ? `Configured (${credentialSource})`
                                : 'Not configured'}
                        </Badge>
                    </Flex>
                    <Flex align="end" gap="2" wrap="wrap">
                        <FormField slotLabel="New credential value" className="flex-1">
                            <TextInput
                                type="password"
                                value={credentialValue}
                                onValueChange={setCredentialValue}
                                autoComplete="off"
                            />
                        </FormField>
                        <Button
                            kind="primary"
                            color="brand"
                            size="small"
                            disabled={
                                !connected ||
                                busy !== null ||
                                !credentialValue ||
                                !CREDENTIAL_PATTERN.test(credential.trim())
                            }
                            onClick={() => void saveCredential()}
                        >
                            Save
                        </Button>
                        <Button
                            kind="tertiary"
                            color="danger"
                            size="small"
                            disabled={
                                !connected ||
                                busy !== null ||
                                !CREDENTIAL_PATTERN.test(credential.trim())
                            }
                            onClick={() => void clearCredential()}
                        >
                            Clear
                        </Button>
                    </Flex>
                </Stack>
            </div>
        </Stack>
    )
}
