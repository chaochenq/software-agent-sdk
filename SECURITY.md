# Security

## Pinned GitHub Actions

Workflow `uses:` refs are pinned to immutable commit SHAs rather than floating
tags. A tag is mutable — whoever controls an action's repository can retarget
`v6` at new code, and every workflow trusting that tag executes it with the
workflow's permissions and secrets on the next run. The trailing comment records
the version the SHA corresponds to.

| Action | Version | Pinned commit |
|---|---|---|
| [actions/checkout](https://github.com/actions/checkout) | v6 | `d23441a48e516b6c34aea4fa41551a30e30af803` |
| [actions/download-artifact](https://github.com/actions/download-artifact) | v8 | `3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c` |
| [actions/github-script](https://github.com/actions/github-script) | v9 | `3a2844b7e9c422d3c10d287c895573f7108da1b3` |
| [actions/setup-node](https://github.com/actions/setup-node) | v6 | `249970729cb0ef3589644e2896645e5dc5ba9c38` |
| [actions/setup-python](https://github.com/actions/setup-python) | v6 | `ece7cb06caefa5fff74198d8649806c4678c61a1` |
| [actions/setup-python](https://github.com/actions/setup-python) | v5 | `a26af69be951a213d495a4c3e4e4022e16d87065` |
| [actions/stale](https://github.com/actions/stale) | v10 | `1e223db275d687790206a7acac4d1a11bd6fe629` |
| [actions/upload-artifact](https://github.com/actions/upload-artifact) | v7 | `043fb46d1a93c77aae656e7c1c64a875d1fc6a0a` |
| [astral-sh/setup-uv](https://github.com/astral-sh/setup-uv) | v7 | `37802adc94f370d6bfd71619e3f0bf239e1f3b78` |
| [tj-actions/changed-files](https://github.com/tj-actions/changed-files) | v47 | `24d32ffd492484c1d75e0c0b894501ddb9d30d62` |

To update one: resolve the new tag with
`git ls-remote https://github.com/<owner>/<repo> 'refs/tags/<tag>^{}'`, replace
the SHA in the workflows and in this table together.
