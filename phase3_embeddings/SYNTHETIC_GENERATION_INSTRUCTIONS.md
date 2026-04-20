# Instructions for Generating Synthetic Training Queries

Create a file `phase3_embeddings/synthetic_queries.py` containing a Python dictionary called `SYNTHETIC_QUERIES`.

For each BNS section listed below, write 3-5 realistic incident descriptions that an Indian citizen might report to the police or search online.

## Rules
1. Write informal, realistic 1-3 sentence incident descriptions in plain English
2. Vary scenarios, victims, locations, and circumstances
3. Do NOT mention section numbers or legal terminology in the queries
4. Each query should clearly map to that specific offense
5. Format as: `SYNTHETIC_QUERIES = { section_number: ["query1", "query2", ...], ... }`

## Example Output Format

```python
SYNTHETIC_QUERIES = {
    303: [
        "Someone stole my bicycle that was parked outside the grocery store",
        "My phone was taken from my desk at work while I was in a meeting",
        "A thief broke into my car and stole the laptop bag from the back seat",
        "My neighbor's kid took my son's cricket bat and refuses to return it",
    ],
    309: [
        "Two men stopped me on the road and took my wallet at knifepoint",
        "I was robbed at gunpoint while walking home from the ATM late at night",
        "A gang of men surrounded my auto-rickshaw and snatched my gold chain by force",
    ],
}
```

## Sections to Cover

Generate for ALL of the following sections (271 total):

### Abetment & Conspiracy (45-62)
- 45: Abetment of a thing
- 46: Abettor
- 47: Abetment in India of offences outside India
- 48: Abetment outside India for offence in India
- 49: Punishment of abetment if act abetted is committed
- 50: Punishment of abetment if person abetted does act with different intention
- 51: Liability of abettor when one act abetted and different act done
- 52: Abettor liable to cumulative punishment
- 53: Liability of abettor for effect different from intended
- 54: Abettor present when offence is committed
- 55: Abetment of offence punishable with death or life imprisonment
- 56: Abetment of offence punishable with imprisonment
- 57: Abetting commission of offence by public or by more than ten persons
- 58: Concealing design to commit offence punishable with death or life imprisonment
- 59: Public servant concealing design to commit offence
- 60: Concealing design to commit offence punishable with imprisonment
- 61: Criminal conspiracy
- 62: Attempting to commit offences punishable with life imprisonment

### Sexual Offences (63-79)
- 63: Rape
- 65: Punishment for rape in certain cases
- 66: Causing death or persistent vegetative state of rape victim
- 67: Sexual intercourse by husband upon wife during separation
- 68: Sexual intercourse by person in authority
- 73: Publishing Court proceedings without permission
- 74: Assault on woman to outrage modesty
- 75: Sexual harassment
- 76: Criminal force on woman with intent to disrobe
- 77: Voyeurism
- 78: Stalking
- 79: Insulting modesty of a woman

### Marriage & Dowry Offences (80-86)
- 80: Dowry death
- 81: Deceitfully inducing belief of lawful marriage
- 82: Bigamy (marrying again during lifetime of spouse)
- 83: Fraudulent marriage ceremony
- 84: Enticing or detaining married woman
- 85: Cruelty by husband or relatives
- 86: Cruelty defined

### Offences Against Children & Unborn (88-99)
- 88: Causing miscarriage
- 89: Causing miscarriage without consent
- 90: Death caused by act to cause miscarriage
- 91: Preventing child from being born alive
- 92: Death of quick unborn child by culpable homicide
- 93: Abandonment of child under 12
- 94: Concealment of birth by secret disposal of body
- 95: Hiring child to commit offence
- 96: Procuration of child
- 97: Kidnapping child under 10 to steal from person
- 98: Selling child for prostitution
- 99: Buying child for prostitution

### Homicide & Bodily Harm (100-125)
- 100: Culpable homicide
- 101: Murder
- 102: Culpable homicide causing death of wrong person
- 103: Punishment for murder
- 104: Murder by life-convict
- 105: Culpable homicide not amounting to murder
- 106: Causing death by negligence
- 107: Abetment of suicide of child or unsound mind person
- 108: Abetment of suicide
- 109: Attempt to murder
- 110: Attempt to commit culpable homicide
- 111: Organised crime
- 112: Petty organised crime
- 113: Terrorist act
- 114: Hurt
- 115: Voluntarily causing hurt
- 116: Grievous hurt
- 117: Voluntarily causing grievous hurt
- 118: Hurt by dangerous weapons
- 119: Hurt to extort property
- 120: Hurt to extort confession
- 121: Hurt to deter public servant
- 122: Hurt on provocation
- 123: Hurt by poison with intent to commit offence
- 124: Grievous hurt by acid
- 125: Endangering life or safety

### Force, Restraint & Assault (126-136)
- 126: Wrongful restraint
- 127: Wrongful confinement
- 128: Force
- 129: Criminal force
- 130: Assault
- 131: Assault or criminal force (not on grave provocation)
- 132: Assault to deter public servant
- 133: Assault to dishonour
- 134: Assault in attempt to steal property from person
- 135: Assault to wrongfully confine
- 136: Assault on grave provocation

### Kidnapping & Trafficking (137-146)
- 137: Kidnapping
- 138: Abduction
- 139: Kidnapping or maiming child for begging
- 144: Exploitation of trafficked person
- 145: Habitual dealing in slaves
- 146: Unlawful compulsory labour

### Offences Against State (147-154)
- 147: Waging war against Government
- 148: Conspiracy to wage war
- 149: Collecting arms to wage war
- 150: Concealing design to wage war
- 151: Assaulting President/Governor to compel
- 152: Endangering sovereignty/unity/integrity of India
- 153: Waging war against foreign State at peace with India
- 154: Depredation on foreign State territories

### Military Offences (164-168)
- 164: Harbouring deserter
- 165: Deserter concealed on merchant vessel
- 166: Abetment of insubordination by soldier
- 167: Persons subject to certain Acts
- 168: Wearing military garb/token

### Election Offences (169-177)
- 169: Candidate, electoral right defined
- 170: Bribery
- 171: Undue influence at elections
- 172: Personation at elections
- 173: Punishment for bribery
- 174: Punishment for undue influence or personation
- 175: False statement in connection with election
- 176: Illegal payments in connection with election
- 177: Failure to keep election accounts

### Counterfeiting (178-182, 188)
- 178: Counterfeiting coin, stamps, currency-notes
- 179: Using forged/counterfeit coin or notes
- 180: Possession of forged coin or notes
- 181: Making instruments for counterfeiting
- 182: Making documents resembling currency
- 188: Taking coining instrument from mint

### Unlawful Assembly & Rioting (189-197)
- 189: Unlawful assembly
- 190: Member of unlawful assembly guilty of offence
- 191: Rioting
- 192: Provoking riot
- 193: Liability of owner of land where riot occurs
- 194: Affray
- 195: Assaulting public servant suppressing riot
- 196: Promoting enmity between groups
- 197: Assertions prejudicial to national integration

### Public Servant Offences (198-212)
- 198: Public servant disobeying law to cause injury
- 199: Public servant disobeying direction under law
- 200: Non-treatment of victim
- 201: Public servant framing incorrect document
- 202: Public servant unlawfully engaging in trade
- 203: Public servant unlawfully buying property
- 204: Personating a public servant
- 205: Wearing garb of public servant with fraudulent intent
- 206: Absconding to avoid summons
- 207: Preventing service of summons
- 208: Non-attendance on order from public servant
- 209: Non-appearance in response to proclamation
- 210: Omission to produce document to public servant
- 211: Omission to give notice to public servant
- 212: Furnishing false information

### Obstruction & Resistance (218-226)
- 218: Resistance to taking property by public servant
- 219: Obstructing sale by public servant
- 220: Illegal purchase of property at public servant's sale
- 221: Obstructing public servant in public functions
- 222: Omission to assist public servant
- 223: Disobedience to public servant's order
- 224: Threat of injury to public servant
- 225: Threat to induce person not to seek protection
- 226: Attempt to commit suicide to compel exercise of lawful power

### False Evidence & Perjury (227-236)
- 227: Giving false evidence
- 228: Fabricating false evidence
- 229: Punishment for false evidence
- 230: False evidence to procure conviction of capital offence
- 231: False evidence to procure conviction of imprisonable offence
- 232: Threatening person to give false evidence
- 233: Using evidence known to be false
- 234: Issuing false certificate
- 235: Using false certificate as true
- 236: False statement in declaration receivable as evidence

### Fraud in Legal Proceedings (242-248)
- 242: False personation in legal proceedings
- 243: Fraudulent concealment of property from seizure
- 244: Fraudulent claim to property to prevent seizure
- 245: Fraudulently suffering decree for sum not due
- 246: Dishonestly making false claim in Court
- 247: Fraudulently obtaining decree for sum not due
- 248: False charge of offence with intent to injure

### Harbouring & Aiding Offenders (252-254)
- 252: Taking gift to recover stolen property
- 253: Harbouring escaped offender
- 254: Harbouring robbers or dacoits

### Corruption of Justice (256-269)
- 256: Public servant framing incorrect record to save person
- 257: Public servant corruptly making report contrary to law
- 258: Confinement by person knowing they act wrongly
- 259: Intentional omission to apprehend
- 260: Intentional omission to apprehend person under sentence
- 261: Escape from custody by public servant's negligence
- 262: Resistance to lawful apprehension (self)
- 263: Resistance to lawful apprehension (another)
- 264: Omission to apprehend (not otherwise provided)
- 265: Resistance to apprehension (not otherwise provided)
- 266: Violation of condition of remission
- 267: Insult to public servant in judicial proceeding
- 268: Personation of assessor
- 269: Failure to appear in Court on bail

### Public Nuisance & Health (270-293)
- 270: Public nuisance
- 271: Negligent act spreading disease
- 272: Malignant act spreading disease
- 273: Disobedience to quarantine rule
- 274: Adulteration of food for sale
- 275: Sale of noxious food or drink
- 276: Adulteration of drugs
- 277: Sale of adulterated drugs
- 278: Sale of drug as different drug
- 279: Fouling water of public spring
- 280: Making atmosphere noxious to health
- 281: Rash driving on public way
- 282: Rash navigation of vessel
- 283: Exhibition of false light/mark/buoy
- 284: Conveying person in unsafe vessel
- 285: Danger or obstruction in public way
- 286: Negligent conduct with poisonous substance
- 287: Negligent conduct with fire or combustible matter
- 288: Negligent conduct with explosive substance
- 289: Negligent conduct with machinery
- 290: Negligent conduct with buildings
- 291: Negligent conduct with animal
- 292: Punishment for public nuisance (not otherwise provided)
- 293: Continuance of nuisance after injunction

### Obscenity & Religion (294-302)
- 294: Sale of obscene books
- 295: Sale of obscene objects to child
- 296: Obscene acts and songs
- 297: Keeping lottery office
- 298: Injuring/defiling place of worship
- 299: Acts intended to outrage religious feelings
- 300: Disturbing religious assembly
- 301: Trespassing on burial places
- 302: Words to wound religious feelings

### Theft & Property (303-314)
- 303: Theft
- 304: Snatching
- 305: Theft in dwelling house, transport, or place of worship
- 306: Theft by clerk or servant
- 307: Theft with preparation for causing death/hurt
- 308: Extortion
- 309: Robbery
- 310: Dacoity
- 311: Robbery/dacoity with attempt to cause death or grievous hurt
- 312: Attempt to commit robbery/dacoity with deadly weapon
- 313: Belonging to gang of robbers
- 314: Dishonest misappropriation of property

### Cheating & Fraud (319-323)
- 319: Cheating by personation
- 320: Fraudulent concealment of property to prevent distribution
- 321: Preventing debt being available for creditors
- 322: Fraudulent deed of transfer with false statement
- 323: Fraudulent removal or concealment of property

### Mischief & Trespass (324-334)
- 324: Mischief
- 325: Mischief by killing or maiming animal
- 326: Mischief by fire, explosive, inundation
- 327: Mischief to rail, aircraft, vessel
- 328: Intentionally running vessel aground
- 329: Criminal trespass and house-trespass
- 330: House-trespass and house-breaking
- 331: Punishment for house-trespass or house-breaking
- 332: House-trespass to commit offence
- 333: House-trespass with preparation for hurt
- 334: Dishonestly breaking open receptacle

### Forgery & Documents (335-344)
- 335: Making a false document
- 336: Forgery
- 337: Forgery of Court record or public register
- 338: Forgery of valuable security or will
- 339: Possession of forged document (s337/338)
- 340: Using forged document as genuine
- 341: Possessing counterfeit seal for forgery
- 342: Counterfeiting authenticating device
- 343: Fraudulent destruction of will or valuable security
- 344: Falsification of accounts

### Property Marks (346-350)
- 346: Tampering with property mark
- 347: Counterfeiting property mark
- 348: Instrument for counterfeiting property mark
- 349: Selling goods with counterfeit property mark
- 350: False mark on receptacle containing goods

### Intimidation & Miscellaneous (351-358)
- 351: Criminal intimidation
- 352: Intentional insult to provoke breach of peace
- 353: Statements conducing to public mischief
- 354: Inducing belief of Divine displeasure
- 355: Misconduct in public by drunken person
- 356: Defamation
- 357: Breach of contract to attend helpless person
- 358: Repeal and savings
