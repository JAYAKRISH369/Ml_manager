import CodeSection from "./CodeSection.jsx";
import {CopyToClipboard} from 'react-copy-to-clipboard';
function CreateSection(props){
return(
    <CopyToClipboard key={props.id} text={props.code}>
    <CodeSection key={props.id} name={props.name} code={props.code}/>
    </CopyToClipboard>
);
}
export default CreateSection;