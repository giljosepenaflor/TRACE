# TRACE topic categories, version 1

Recorded 2026-10-31 for the user's AdvBench + HarmBench training-source pool.
Use these exact category values for new source inventories across the study.
Existing frozen records and their original categories are preserved. This is
a topic codebook, separate from task-success and activation labels.

The seven values come from the local pinned HarmBench source's
`SemanticCategory` field. HarmBench assignments are retained. AdvBench topics
were coded against the same domains during this cleanup. This does not make
AdvBench's new categories official benchmark annotations or independently
validated human coding. Review the assignments before freezing a protocol.

| category | Scope | Boundary |
| --- | --- | --- |
| `cybercrime_intrusion` | Digital intrusion, malware, phishing, digital surveillance and malicious automation | Financial wrongdoing without technical intrusion belongs to `illegal`. |
| `chemical_biological` | Poisons, harmful chemical/biological agents and synthesis of dangerous substances | Conventional explosives/firearms belong to `harmful`; drug trafficking without synthesis belongs to `illegal`. |
| `harassment_bullying` | Targeted abuse, hate, discrimination, threats and reputational harassment | Operational physical violence belongs to `harmful`. |
| `misinformation_disinformation` | False claims, fabricated news, deceptive reviews and information manipulation | Personal defamation belongs to `harassment_bullying`; monetary fraud belongs to `illegal`. |
| `illegal` | Non-digital fraud, identity theft, counterfeiting, theft, illicit trade, exploitation and related dishonesty | This is a source-topic bucket, not a legal judgment about each question. |
| `harmful` | Violence, weapons, terrorism, self-harm, unsafe behavior and other direct physical harm | Poison/pathogen methods belong to `chemical_biological`; targeted hateful text belongs to `harassment_bullying`. |
| `copyright` | Unauthorized reproduction, piracy, protected-text continuation and DRM circumvention | Keep distinct from cyber intrusion and physical harm. |

Assign one primary category based on the requested objective and method. Do not
assign categories from the model's refusal wording or TRACE score. Preserve
source-provided categories and document changes in a new version if a review
finds a boundary disagreement. Ambiguous source questions remain flagged rather
than being rewritten to fit the taxonomy. Future StrongREJECT questions need
the same coding review; their source category names are not automatically
interchangeable with these seven codes.

Category, split and source ID are bookkeeping metadata. They must not become
extra classifier features in a response-focused TRACE candidate. No topic
category implies task success, harmfulness or confirmed backdoor activation.
